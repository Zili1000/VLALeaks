
import itertools
import os
from rouge import Rouge
import wandb
os.environ['WANDB_API_KEY'] = ''  # Set the wandb API key
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, IterableDataset
import draccus 
import torch
import torch.distributed as dist
from tqdm import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import random
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction 
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, MaskedRLDSBatchTransform, NonMemberMaskTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
import numpy as np
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import torch.nn.functional as F
from collections import defaultdict
import zlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional




class AdvancedAttentionFeatures:
    def __init__(self, num_image_tokens: int = 256, num_action_tokens: Optional[int] = None):
        """
        Args:
            num_image_tokens: The number of image tokens (if known), used to separate image and text regions in the attention matrix
            num_action_tokens: The number of action tokens (if known), otherwise inferred from the attention matrix
        """
        self.num_image_tokens = num_image_tokens
        self.num_action_tokens = num_action_tokens
        
    def compute_attention_entropy(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """Calculate the entropy of the attention distribution"""
        entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-10), dim=-1)
        return entropy.mean(dim=-1)
    
    def compute_attention_concentration(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """Calculate the concentration of the attention distribution (proportion of maximum attention weight)"""
        max_attn = attention_matrix.max(dim=-1)[0]
        return max_attn.mean(dim=-1)
    
    def compute_frobenius_distance(
        self, 
        attention_matrix1: torch.Tensor, 
        attention_matrix2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the Frobenius distance between two attention matrices
        
        Args:
            attention_matrix1, attention_matrix2: (B, H, S, S) or (B, S, S)
        """
        # If the input is 4 dimensions (including the header dimension), flatten the header dimension for distance calculation
        if attention_matrix1.dim() == 4:
            B, H, S1, S2 = attention_matrix1.shape
            attn1_flat = attention_matrix1.view(B, -1, S1, S2)
            attn2_flat = attention_matrix2.view(B, -1, S1, S2)
            distance = torch.norm(attn1_flat - attn2_flat, p='fro', dim=(1,2,3))
        else:
            distance = torch.norm(attention_matrix1 - attention_matrix2, p='fro', dim=(1,2))
        return distance
    
    def compute_kl_divergence(
        self, 
        attention_matrix1: torch.Tensor, 
        attention_matrix2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the KL divergence between two attention distributions (symmetrized version)
        """
        # Ensure numerical stability
        eps = 1e-10
        
        # Compute KL divergence for each token's attention distribution
        if attention_matrix1.dim() == 4:
            B, H, S, _ = attention_matrix1.shape
            # Flatten the header dimension for KL calculation
            attn1 = attention_matrix1.view(B, -1, S, S)
            attn2 = attention_matrix2.view(B, -1, S, S)
        else:
            attn1 = attention_matrix1.unsqueeze(1)
            attn2 = attention_matrix2.unsqueeze(1)
            B, _, S, _ = attn1.shape
        
        # Calculate KL divergence D(P||Q) = sum P * log(P/Q)
        kl_pq = torch.sum(attn1 * (torch.log(attn1 + eps) - torch.log(attn2 + eps)), dim=(1,2,3))
        # Calculate KL divergence D(Q||P)
        kl_qp = torch.sum(attn2 * (torch.log(attn2 + eps) - torch.log(attn1 + eps)), dim=(1,2,3))
        # Return the symmetric KL divergence
        return (kl_pq + kl_qp) / 2
    
    def compute_cross_modal_flow(self, attentions: tuple) -> Dict[str, torch.Tensor]:
        """
        Calculate cross-modal information flow features based on the attention matrices across layers
        """
        features = {}
        
        # Collect image-to-text attention from all layers
        img_to_text_attentions = []
        for layer_attn in attentions:
            avg_attn = layer_attn.mean(dim=1)  # (B, S, S)
            img_to_text = avg_attn[:, :self.num_image_tokens, self.num_image_tokens:]
            img_to_text_attentions.append(img_to_text)
        
        # 1. The temporal variation of the attention flow (interlayer difference)
        for i in range(len(img_to_text_attentions) - 1):
            flow_change = img_to_text_attentions[i+1] - img_to_text_attentions[i]
            features[f'flow_change_{i}'] = flow_change.mean(dim=(1,2))
        
        # 2. The final layer vs. the initial layer comparison        features['final_layer_attention'] = img_to_text_attentions[-1].mean(dim=(1,2))
        features['final_vs_initial'] = (
            img_to_text_attentions[-1] - img_to_text_attentions[0]
        ).mean(dim=(1,2))
        
        # 3. Add Frobenius distance
        features['frobenius_distance_first_last'] = self.compute_frobenius_distance(
            img_to_text_attentions[0], img_to_text_attentions[-1]
        )
        
        # 4. Add KL divergence
        features['kl_divergence_first_last'] = self.compute_kl_divergence(
            img_to_text_attentions[0], img_to_text_attentions[-1]
        )
        
        return features
    
    def extract_action_features(
        self, 
        attentions: tuple, 
        # action_token_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention features related to actions
        
        Args:
            attentions: Tuple of attention matrices
            action_token_indices: List of positions for action tokens, assumed to be at the end of the sequence if None
        """
        features = {}
        
        action_token_indices = list(range(self.num_action_tokens - 7, self.num_action_tokens))
        # If the position of the action token is not specified, assume it is after the text area
        # if action_token_indices is None and self.num_action_tokens is not None:
        #     text_end = attentions[0].shape[-1]
        #     action_token_indices = list(range(text_end - self.num_action_tokens, text_end))
        
        # if action_token_indices is None:
        #     # The position of the action token cannot be determined, and an empty feature is returned
        #     return features
        
        for layer_idx, layer_attn in enumerate(attentions):
            avg_attn = layer_attn.mean(dim=1)  # (B, S, S)
            B, S, _ = avg_attn.shape
            
            # Extract attention regions related to actions
            action_attn = avg_attn[:, action_token_indices, :]  # (B, num_actions, S)
            
            # 1. Attention distribution of actions towards images and text
            action_to_img = action_attn[:, :, :self.num_image_tokens]
            action_to_text = action_attn[:, :, self.num_image_tokens:]
            
            features.update({
                f'layer_{layer_idx}_action_to_img_mean': action_to_img.mean(dim=(1,2)),
                f'layer_{layer_idx}_action_to_img_std': action_to_img.std(dim=(1,2)),
                f'layer_{layer_idx}_action_to_text_mean': action_to_text.mean(dim=(1,2)),
                f'layer_{layer_idx}_action_to_text_std': action_to_text.std(dim=(1,2)),
                f'layer_{layer_idx}_action_entropy': self.compute_attention_entropy(action_attn),
                f'layer_{layer_idx}_action_concentration': self.compute_attention_concentration(action_attn),
            })
            
            # 2. Attention from images/text to actions
            img_to_action = avg_attn[:, :self.num_image_tokens, action_token_indices]
            text_to_action = avg_attn[:, self.num_image_tokens:, action_token_indices]
            
            features.update({
                f'layer_{layer_idx}_img_to_action_mean': img_to_action.mean(dim=(1,2)),
                f'layer_{layer_idx}_text_to_action_mean': text_to_action.mean(dim=(1,2)),
            })
            
            # 3. Intra-action attention (if multiple action tokens)
            if len(action_token_indices) > 1:
                action_intra = avg_attn[:, action_token_indices, :][:, :, action_token_indices]
                features.update({
                    f'layer_{layer_idx}_action_intra_mean': action_intra.mean(dim=(1,2)),
                    f'layer_{layer_idx}_action_intra_std': action_intra.std(dim=(1,2)),
                })
        
        return features
    
    def extract_features(
        self, 
        attentions: tuple,
        # action_token_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive feature extraction (including action-related features)
        """
        features = {}
        batch_size = attentions[0].shape[0]
        
        # Store features from all layers
        layer_features = []
        
        for layer_idx, layer_attn in enumerate(attentions):
            # Average of all heads
            avg_attn = layer_attn.mean(dim=1)  # (B, S, S)
            
            # Determine the end position of the text region (assuming actions are after the text)
            text_end = self.num_action_tokens - 7
            
            # Separate different regions
            img_region = avg_attn[:, :self.num_image_tokens, :self.num_image_tokens]
            img_to_text = avg_attn[:, :self.num_image_tokens, self.num_image_tokens:text_end]
            text_to_img = avg_attn[:, self.num_image_tokens:text_end, :self.num_image_tokens]
            text_region = avg_attn[:, self.num_image_tokens:text_end, self.num_image_tokens:text_end]
            
            
            # Calculate basic statistics for each region
            layer_feat = {
                'img_intra_mean': img_region.mean(dim=(1,2)),
                'img_intra_std': img_region.std(dim=(1,2)),
                'img_intra_entropy': self.compute_attention_entropy(img_region),
                
                'img_to_text_mean': img_to_text.mean(dim=(1,2)),
                'img_to_text_std': img_to_text.std(dim=(1,2)),
                'img_to_text_entropy': self.compute_attention_entropy(img_to_text),
                'img_to_text_concentration': self.compute_attention_concentration(img_to_text),
                
                'text_to_img_mean': text_to_img.mean(dim=(1,2)),
                'text_to_img_std': text_to_img.std(dim=(1,2)),
                'text_to_img_entropy': self.compute_attention_entropy(text_to_img),
                'text_to_img_concentration': self.compute_attention_concentration(text_to_img),

                'text_intra_mean': text_region.mean(dim=(1,2)),
                'text_intra_std': text_region.std(dim=(1,2)),
                'text_intra_entropy': self.compute_attention_entropy(text_region),
            }
            
            # Add layer index to feature names and store features
            for key, value in layer_feat.items():
                features[f'layer_{layer_idx}_{key}'] = value
            
            layer_features.append(layer_feat)
        
        # Add cross-layer statistical features
        # 1. The inter-layer variation of attention concentration for image-to-text attention
        img_to_text_conc = torch.stack([
            f['img_to_text_concentration'] for f in layer_features
        ])
        features['img_to_text_concentration_trend'] = img_to_text_conc.mean(dim=0)
        features['img_to_text_concentration_std'] = img_to_text_conc.std(dim=0)
        
        # 2. Inter-layer consistency of cross-modal interaction (using Frobenius distance)
        img_to_text_list = [
            attentions[layer_idx].mean(dim=1)[:, :self.num_image_tokens, self.num_image_tokens:]
            for layer_idx in range(len(attentions))
        ]
        
        # Calculate Frobenius distances between adjacent layers
        for i in range(len(img_to_text_list) - 1):
            features[f'frobenius_distance_layer_{i}_to_{i+1}'] = self.compute_frobenius_distance(
                img_to_text_list[i], img_to_text_list[i+1]
            )
        
        # 3. Calculate KL divergences between adjacent layers
        for i in range(len(img_to_text_list) - 1):
            features[f'kl_divergence_layer_{i}_to_{i+1}'] = self.compute_kl_divergence(
                img_to_text_list[i], img_to_text_list[i+1]
            )
        
        # 4. Intra-modal vs. cross-modal attention comparison
        for layer_idx, f in enumerate(layer_features):
            intra_vs_cross = (
                f['img_intra_mean'] + f['text_intra_mean'] - 
                f['img_to_text_mean'] - f['text_to_img_mean']
            ) / 2
            features[f'layer_{layer_idx}_intra_vs_cross'] = intra_vs_cross
        
        # 5. Add action-related features
        action_features = self.extract_action_features(attentions)
        features.update(action_features)
        
        return features
    
    def extract_pairwise_features(
        self, 
        attentions1: tuple, 
        attentions2: tuple,
        action_token_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract contrastive features between two samples (for contrastive learning or similarity tasks)
        """
        features = {}
        
        # Extract attention matrices for both samples and calculate pairwise distance metrics
        for layer_idx in range(len(attentions1)):
            attn1 = attentions1[layer_idx].mean(dim=1)
            attn2 = attentions2[layer_idx].mean(dim=1)
            
            # Calculate pairwise distance metrics
            features[f'layer_{layer_idx}_frobenius_distance'] = self.compute_frobenius_distance(attn1, attn2)
            features[f'layer_{layer_idx}_kl_divergence'] = self.compute_kl_divergence(attn1, attn2)
            
            # Calculate image region differences
            img1 = attn1[:, :self.num_image_tokens, :self.num_image_tokens]
            img2 = attn2[:, :self.num_image_tokens, :self.num_image_tokens]
            features[f'layer_{layer_idx}_img_frobenius'] = self.compute_frobenius_distance(img1, img2)
            
            # Calculate cross-modal region differences
            cross1 = attn1[:, :self.num_image_tokens, self.num_image_tokens:]
            cross2 = attn2[:, :self.num_image_tokens, self.num_image_tokens:]
            features[f'layer_{layer_idx}_cross_frobenius'] = self.compute_frobenius_distance(cross1, cross2)
            
            # If action tokens are available, calculate action region differences
            if action_token_indices is not None:
                action1 = attn1[:, action_token_indices, :]
                action2 = attn2[:, action_token_indices, :]
                features[f'layer_{layer_idx}_action_frobenius'] = self.compute_frobenius_distance(action1, action2)
        
        return features


class ScoresLogger:
    def __init__(self, log_dir='/home/tcs/4t01/lxk/openvla/log', filename=None):
        os.makedirs(log_dir, exist_ok=True)
        
        if filename is None:
            filename = f'scores_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        self.filepath = os.path.join(log_dir, filename)
        self.scores_data = defaultdict(list)  # Used for temporary data storage
        
    def log_scores(self, scores):
        """
        Record all the values in the scores dictionary and store them in wide format
        scores: defaultdict(list) 格式，如 {'mink_0.4': [-0.002032], 'mink_0.5': [-0.001357], ...}
        """
        metric_names = sorted(scores.keys())

        max_length = max(len(values) for values in scores.values())
        
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(metric_names)
            
            for i in range(max_length):
                row = []
                for metric_name in metric_names:
                    values = scores[metric_name]
                    if i < len(values):
                        value = values[i]
                        if isinstance(value, float) and np.isnan(value):
                            row.append('nan')
                        else:
                            row.append(value)
                    else:
                        row.append('')
                writer.writerow(row)
        
        print(f"Scores saved to {self.filepath} in wide format")
    
    def log_scores_with_ratios(self, scores):

        mink_data = {}
        minkpp_data = {}
        
        for metric_name, values in scores.items():
            if metric_name.startswith('mink++'):
                minkpp_data[metric_name] = values
            elif metric_name.startswith('mink'):
                mink_data[metric_name] = values
        
        if mink_data:
            self._save_category('mink', mink_data)
        
        if minkpp_data:
            self._save_category('mink++', minkpp_data)
    
    def _save_category(self, category_name, data):
        category_filepath = self.filepath.replace('.csv', f'_{category_name}.csv')
        
        metric_names = sorted(data.keys())
        
        max_length = max(len(values) for values in data.values())
        
        with open(category_filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(metric_names)
            
            for i in range(max_length):
                row = []
                for metric_name in metric_names:
                    values = data[metric_name]
                    if i < len(values):
                        value = values[i]
                        if isinstance(value, float) and np.isnan(value):
                            row.append('nan')
                        else:
                            row.append(value)
                    else:
                        row.append('')
                writer.writerow(row)
        
        print(f"{category_name} scores saved to {category_filepath}")
    
    def log_scores_transposed(self, scores):

        metric_names = sorted(scores.keys())
        
        max_length = max(len(values) for values in scores.values())
        
        with open(self.filepath.replace('.csv', '_transposed.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            
            header = ['metric_name'] + [f'value_{i+1}' for i in range(max_length)]
            writer.writerow(header)
            
            for metric_name in metric_names:
                row = [metric_name]
                values = scores[metric_name]
                
                for i in range(max_length):
                    if i < len(values):
                        value = values[i]
                        if isinstance(value, float) and np.isnan(value):
                            row.append('nan')
                        else:
                            row.append(value)
                    else:
                        row.append('')
                
                writer.writerow(row)
        
        print(f"Transposed scores saved to {self.filepath.replace('.csv', '_transposed.csv')}")


class MetricsLogger:
    def __init__(self, log_dir='/home/tcs/4t01/lxk/openvla/log', filename=None):
        os.makedirs(log_dir, exist_ok=True)
        
        if filename is None:
            from datetime import datetime
            filename = f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        self.filepath = os.path.join(log_dir, filename)

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'action_accuracy', 'l1_loss'])
        
        self.step = 0
    
    def log(self, loss, accuracy, l1_loss):

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.step, loss, accuracy, l1_loss])
        
        self.step += 1
    
    def log_batch(self, recent_losses, recent_action_accuracies, recent_l1_losses):

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, (loss, acc, l1) in enumerate(zip(recent_losses, recent_action_accuracies, recent_l1_losses)):
                writer.writerow([self.step + i, loss, acc, l1])
        
        self.step += len(recent_losses)

def set_seed(seed=42):
    """Set all random seeds to ensure repeatability"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

@dataclass
class FinetuneConfig:
    """Fine-tune the configuration class - all configurable parameters"""
    # vla_path: str = "openvla/openvla-7b"
    vla_path: str = "/home/tcs/4t01/lxk/openvla/log/openvla-7b+libero_spatial_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0"

    # path
    data_root_dir: Path = Path("/home/tcs/4t01/lxk/datasets/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path("/home/tcs/4t01/lxk/openvla/log")     # The directory path for storing logs and checkpoints
    adapter_tmp_dir: Path = Path("/home/tcs/4t01/lxk/model/openvla")                   

    # Tuning Parameter
    batch_size: int = 1
    max_steps: int = 200_000
    # max_steps: int = 5000
    save_steps: int = 500
    # save_steps: int = 5000
    learning_rate: float = 5e-4 
    grad_accumulation_steps: int = 1
    image_aug: bool = False
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True

    # LoRA Parameter
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    # wandb_project: str = "openvla"
    # wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None


@draccus.wrap()
def extraction(cfg: FinetuneConfig) -> None:
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()  # Obtain the distributed state
    torch.cuda.set_device(device_id := distributed_state.local_process_index)  # Set the GPU used by the current process
    torch.cuda.empty_cache()

    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"  
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        output_attentions=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group("nccl", rank=0, world_size=1)
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    
    # Create a batch converter to convert RLDS batches into model input formats
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        # split_ratio=0.8,
        # is_train_split=True,
    )

    test_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        # split_ratio=0.8,
        # is_train_split=False,
    )


    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    test_dataset_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )


    # Member! - Training set evaluation cycle
    vla.eval()  # Set to evaluation mode
    save_path = '/home/tcs/4t01/lxk/openvla/log/member/attention_features.pt'
    features_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Traverse the data")):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_attentions=True, 
                )
                loss = output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps

            # Access the attention matrix
            if hasattr(output, 'attentions') and output.attentions is not None:
                attentions = output.attentions  # tuple of (batch_size, num_heads, seq_len, seq_len) per layer

            logits = output.logits
            num_image_tokens = vla.module.vision_backbone.featurizer.patch_embed.num_patches
            action_logits = output.logits[:, num_image_tokens : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device) 
            mask = action_gt > action_tokenizer.action_token_begin_idx

            image_logits = logits[:, :num_image_tokens][0]
            instruction_logits = action_logits[~mask]
            pin7_action_logits = action_logits[mask]

            instruction_logits_len = len(instruction_logits)

            attentionfeatures = AdvancedAttentionFeatures(num_image_tokens,num_image_tokens+instruction_logits_len+7)
            aaa = attentionfeatures.extract_features(attentions)

            feature_vector = torch.cat([v.flatten() for v in aaa.values()]).cpu()
            features_list.append(feature_vector)
            
            if (batch_idx + 1) % 100 == 0:
                if Path(save_path).exists():
                    existing = torch.load(save_path)
                    all_features = existing + features_list
                else:
                    all_features = features_list
                torch.save(all_features, save_path)
                
                features_list = []
            if batch_idx >= 10000:
                break

    # Non-member! - Training set evaluation cycle
    vla.eval()  # Set to evaluation mode
    save_path = '/home/tcs/4t01/lxk/openvla/log/nonmember/attention_features.pt'
    features_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataset_dataloader, desc="Traverse the data")):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_attentions=True, 
                )
                loss = output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps

            # Access the attention matrix
            if hasattr(output, 'attentions') and output.attentions is not None:
                attentions = output.attentions  # tuple of (batch_size, num_heads, seq_len, seq_len) per layer

            logits = output.logits
            num_image_tokens = vla.module.vision_backbone.featurizer.patch_embed.num_patches
            action_logits = output.logits[:, num_image_tokens : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device) 
            mask = action_gt > action_tokenizer.action_token_begin_idx

            image_logits = logits[:, :num_image_tokens][0]
            instruction_logits = action_logits[~mask]
            pin7_action_logits = action_logits[mask]

            instruction_logits_len = len(instruction_logits)

            attentionfeatures = AdvancedAttentionFeatures(num_image_tokens,num_image_tokens+instruction_logits_len+7)
            aaa = attentionfeatures.extract_features(attentions)

            feature_vector = torch.cat([v.flatten() for v in aaa.values()]).cpu()
            features_list.append(feature_vector)
            
            if (batch_idx + 1) % 100 == 0:
                if Path(save_path).exists():
                    existing = torch.load(save_path)
                    all_features = existing + features_list
                else:
                    all_features = features_list
                torch.save(all_features, save_path)
                
                features_list = []
            if batch_idx >= 10000:
                break

if __name__ == "__main__":
    set_seed(42)  # Set random seeds
    extraction()  # Run feature extraction function
    