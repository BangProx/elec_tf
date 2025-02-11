import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
import math
import os
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader  # 이 부분이 중요합니다


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, 1)  # 출력 차원을 1로 설정
        
        self.d_model = d_model

    def forward(self, src, tgt):
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Create mask for decoder
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer decoder
        output = self.transformer_decoder(tgt, src, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_layer(output)
        output = output.squeeze(-1)  # 마지막 차원 제거 (N, L, 1) -> (N, L)
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PowerDemandPredictor:
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerDecoderModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
    def train(self, train_loader, val_loader, epochs=100):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        print("Starting training...")
        print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Train MAPE':>12} {'Val MAPE':>12}")
        print("="*60)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_actuals = []
            
            for batch_idx, (src, tgt, y) in enumerate(train_loader):
                src, tgt, y = src.to(self.device), tgt.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(src, tgt)
                loss = self.criterion(output, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Store predictions and actuals for metrics
                train_predictions.extend(output.detach().cpu().numpy())
                train_actuals.extend(y.detach().cpu().numpy())
            
            # Calculate training metrics
            train_metrics_dict = self.calculate_metrics(
                np.array(train_actuals), 
                np.array(train_predictions)
            )
            
            # Validation loop
            val_loss, val_metrics_dict = self.evaluate(val_loader)  # 두 값을 받음
            
            # Store losses and metrics
            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss)
            train_metrics.append(train_metrics_dict)
            val_metrics.append(val_metrics_dict)
            
            # Print progress
            print(f"{epoch:5d} {train_losses[-1]:12.6f} {val_loss:12.6f} "
                  f"{train_metrics_dict['MAPE']:12.2f} {val_metrics_dict['MAPE']:12.2f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                print(f"Epoch {epoch}: New best model saved (Val Loss: {val_loss:.6f})")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for src, tgt, y in data_loader:
                src, tgt, y = src.to(self.device), tgt.to(self.device), y.to(self.device)
                output = self.model(src, tgt)
                loss = self.criterion(output, y)
                total_loss += loss.item()
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(y.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(np.array(actuals), np.array(predictions))
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, metrics  # 두 값을 반환하도록 수정
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate multiple evaluation metrics"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R-squared': r2
        }
    
    def predict(self, src, tgt):
        self.model.eval()
        with torch.no_grad():
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            output = self.model(src, tgt)
        return output.cpu().numpy()
    
    def save_model(self, path):
        """Save model weights"""
        os.makedirs('models', exist_ok=True)
        full_path = os.path.join('models', path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, full_path)
        
    def load_model(self, path):
        """Load model weights"""
        full_path = os.path.join('models', path)
        checkpoint = torch.load(full_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def create_sequences(data, seq_length):
    """시퀀스 데이터 생성"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        # 입력 시퀀스
        seq = data[i:i + seq_length]
        # 타겟 (다음 24시간의 전력 수요)
        target = data[i + seq_length:i + seq_length + 24, 0]  # Hourly Sum만 타겟으로 사용
        
        if len(target) == 24:  # 타겟이 24시간치가 있는 경우만 사용
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

class PowerDemandDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.sequences[idx, -24:],  # decoder input (last 24 hours)
            self.targets[idx]
        )

def prepare_data(processed_de, seq_length=168, batch_size=32):  # 1주일(168시간) 기준
    # numpy array로 변환
    data = processed_de.values
    
    # 시퀀스 생성
    sequences, targets = create_sequences(data, seq_length)
    
    # 학습/검증/테스트 분할 (70/15/15)
    train_seq, temp_seq, train_target, temp_target = train_test_split(
        sequences, targets, test_size=0.3, random_state=42
    )
    val_seq, test_seq, val_target, test_target = train_test_split(
        temp_seq, temp_target, test_size=0.5, random_state=42
    )
    
    # Dataset 생성
    train_dataset = PowerDemandDataset(train_seq, train_target)
    val_dataset = PowerDemandDataset(val_seq, val_target)
    test_dataset = PowerDemandDataset(test_seq, test_target)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, (train_seq, val_seq, test_seq)

def train_and_evaluate(model_predictor, train_loader, val_loader, test_loader, scaler, epochs=100):
    """모델 학습 및 평가"""
    # 모델 학습
    model_predictor.train(train_loader, val_loader, epochs)
    
    # 테스트 세트에서 성능 평가
    model_predictor.model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for src, tgt, y in test_loader:
            src = src.to(model_predictor.device)
            tgt = tgt.to(model_predictor.device)
            output = model_predictor.predict(src, tgt)
            
            # 원래 스케일로 복원
            pred_reshaped = output.reshape(-1, 1)
            actual_reshaped = y.numpy().reshape(-1, 1)
            
            pred_original = scaler.inverse_transform(pred_reshaped)
            actual_original = scaler.inverse_transform(actual_reshaped)
            
            predictions.extend(pred_original.flatten())
            actuals.extend(actual_original.flatten())
    
    # 평가 지표 계산
    metrics = model_predictor.calculate_metrics(np.array(actuals), np.array(predictions))
    
    print("\nTest Set Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return predictions, actuals, metrics

def scale_in_chunks(scaler, data, chunk_size):
    scaled_data = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_reshaped = chunk.reshape(-1, 1)
        if i == 0 or len(chunk) == chunk_size:
            scaled_chunk = scaler.fit_transform(chunk_reshaped)
        else:
            scaled_chunk = scaler.transform(chunk_reshaped)
        scaled_data.extend(scaled_chunk.flatten())
    return np.array(scaled_data)


def plot_training_progress(history):
    """학습 과정 시각화"""
    
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # MAPE plot
    plt.subplot(2, 2, 2)
    train_mape = [m['MAPE'] for m in history['train_metrics']]
    val_mape = [m['MAPE'] for m in history['val_metrics']]
    plt.plot(train_mape, label='Train MAPE')
    plt.plot(val_mape, label='Val MAPE')
    plt.title('Training and Validation MAPE')
    plt.legend()
    
    # R-squared plot
    plt.subplot(2, 2, 3)
    train_r2 = [m['R-squared'] for m in history['train_metrics']]
    val_r2 = [m['R-squared'] for m in history['val_metrics']]
    plt.plot(train_r2, label='Train R²')
    plt.plot(val_r2, label='Val R²')
    plt.title('Training and Validation R²')
    plt.legend()
    
    # RMSE plot
    plt.subplot(2, 2, 4)
    train_rmse = [m['RMSE'] for m in history['train_metrics']]
    val_rmse = [m['RMSE'] for m in history['val_metrics']]
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse, label='Val RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 메인 실행 코드 수정
def main():

    de = pd.read_csv('DETotal.csv')
    de['Hourly Time'] = pd.to_datetime(de['Hourly Time'], format='%d.%m.%Y %H:%M')
    de['weekday'] = de['Hourly Time'].dt.dayofweek  
    de['hour'] = de['Hourly Time'].dt.hour         
    de['month'] = de['Hourly Time'].dt.month

    # 2. StandardScaler 정의 및 데이터 스케일링
    
    scaler = StandardScaler()
    hourly_sum = np.array(de['Hourly Sum'])
    month = np.array(de['month'])
    hour = np.array(de['hour'])

    scaled_hourly_sum = scale_in_chunks(scaler, hourly_sum, 24)
    scaled_month = scale_in_chunks(scaler, month, 24)
    scaled_hour = scale_in_chunks(scaler, hour, 24)

    de['Hourly Sum'] = scaled_hourly_sum
    de['month'] = scaled_month
    de['hour'] = scaled_hour

    # 3. 처리된 데이터프레임 생성
    processed_de = de[['Hourly Sum', 'month', 'hour']]

    # 데이터 준비
    seq_length = 168
    batch_size = 32
    train_loader, val_loader, test_loader, (train_seq, val_seq, test_seq) = prepare_data(
        processed_de, seq_length, batch_size
    )
    
    # 모델 초기화
    input_dim = 3
    model_predictor = PowerDemandPredictor(
        input_dim=input_dim,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1
    )
    
    # 학습 실행 및 히스토리 저장
    history = model_predictor.train(train_loader, val_loader, epochs=100)
    
    # 학습 과정 시각화
    plot_training_progress(history)
    
    # 테스트 세트 최종 평가
    test_loss, test_metrics = model_predictor.evaluate(test_loader)
    
    print("\nFinal Test Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 모델 저장
    model_predictor.save_model('power_demand_model.pth')
    
    return history, test_metrics


if __name__ == "__main__":
    predictions, actuals, metrics = main()