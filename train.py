import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

from dataset import S3DISDataset
from pointnet2 import PointNet2Seg
from metrics import calculate_metrics, calculate_iou


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (points, labels) in enumerate(pbar):
        points = points.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(points)  # [B, N, num_classes]
        
        # Reshape для loss
        logits = logits.reshape(-1, logits.shape[-1])  # [B*N, num_classes]
        labels = labels.reshape(-1)  # [B*N]
        
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Метрики
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.size(0)
        total_loss += loss.item()
        
        # Обновляем прогресс-бар
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * total_correct / total_points:.2f}%'
        })
        
        # Логируем в TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy', 100.0 * total_correct / total_points, global_step)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * total_correct / total_points
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, num_classes):
    """Валидация модели"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in tqdm(val_loader, desc='Validation'):
            points = points.to(device)
            labels = labels.to(device)
            
            logits = model(points)
            logits = logits.reshape(-1, logits.shape[-1])
            labels_flat = labels.reshape(-1)
            
            loss = criterion(logits, labels_flat)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels_flat).sum().item()
            total_points += labels_flat.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_flat.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * total_correct / total_points
    
    # Вычисляем IoU
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    iou = calculate_iou(all_preds, all_labels, num_classes)
    mean_iou = np.mean([iou[i] for i in range(num_classes) if iou[i] is not None])
    
    return avg_loss, avg_acc, mean_iou


def main():
    parser = argparse.ArgumentParser(description='Train PointNet++ for Semantic Segmentation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to S3DIS dataset')
    parser.add_argument('--area', type=str, default='Area_1', help='Area to use (Area_1, Area_2, etc.)')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    
    args = parser.parse_args()
    
    # Создаем директории
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Устройство
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Датасеты
    print('Loading datasets...')
    train_dataset = S3DISDataset(
        args.data_dir, area=args.area, split='train',
        num_points=args.num_points
    )
    val_dataset = S3DISDataset(
        args.data_dir, area=args.area, split='val',
        num_points=args.num_points
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Модель
    print('Initializing model...')
    model = PointNet2Seg(num_classes=args.num_classes, num_points=args.num_points).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Функция потерь с весами классов (для дисбаланса)
    # Вычисляем веса на основе частоты классов в обучающей выборке
    class_weights = torch.ones(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Обучение
    best_iou = 0.0
    print('Starting training...')
    
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 50)
        
        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Валидация
        val_loss, val_acc, val_iou = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        
        # Обновляем learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Логируем
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
        writer.add_scalar('Epoch/Val_IoU', val_iou, epoch)
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val IoU: {val_iou:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Сохраняем лучшую модель
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'✓ Saved best model (IoU: {best_iou:.4f})')
        
        # Сохраняем последнюю модель
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'val_acc': val_acc,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'last_model.pth'))
    
    writer.close()
    print('\nTraining completed!')
    print(f'Best IoU: {best_iou:.4f}')


if __name__ == '__main__':
    main()

