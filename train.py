import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===== 1. 데이터셋 준비 =====
data = [
    {"text": "노래 듣고 싶어", "labels": ["Melon", "Spotify", "벅스", "Youtube"]},
    {"text": "오늘 비 오나?", "labels": ["날씨"]},
    {"text": "오늘 뭐 먹지?", "labels": ["냉장고 관리", "만개의 레시피", "식단 플래너"]},
    {"text": "일정 알려줘", "labels": ["캘린더", "식단 플래너"]},
    {"text": "할 일 적어줘", "labels": ["메모", "화이트보드"]},
    {"text": "쇼핑 목록 추가해줘", "labels": ["쇼핑 리스트"]},
    {"text": "내일 점심 뭐 먹을까?", "labels": ["식단 플래너", "만개의 레시피", "냉장고 관리"]},
    {"text": "알람 맞춰줘", "labels": ["타이머", "빅스비"]},
    {"text": "사진 보여줘", "labels": ["갤러리", "Smart View"]},
    {"text": "캠핑 사진 TV로 틀어줘", "labels": ["Samsung TV Plus", "Smart View", "갤러리"]},
    {"text": "인터넷 열어줘", "labels": ["인터넷"]},
    {"text": "요즘 유행하는 음악 틀어줘", "labels": ["Melon", "Spotify", "Youtube", "벅스"]},
    {"text": "식중독 주의할 음식 뭐야?", "labels": ["위해식품 알리미"]},
    {"text": "화면 녹화 어케해?", "labels": ["설정"]},
    {"text": "TV에서 영화 틀어줘", "labels": ["Samsung TV Plus", "Youtube"]},
    {"text": "오늘 뭐 해야 하지?", "labels": ["캘린더", "메모", "식단 플래너"]},
    {"text": "AI로 물건 인식할 수 있어?", "labels": ["AI 비전 인사이드"]},
    {"text": "배경화면 바꾸고 싶어", "labels": ["설정"]},
]

# ===== 2. 멀티라벨 클래스 사전 생성 =====
all_labels = set()
for item in data:
    all_labels.update(item['labels'])
label_list = sorted(all_labels)
label_to_idx = {label: i for i, label in enumerate(label_list)}

# ===== 3. 토큰화 및 단어 집합 생성 (매우 간단한 방법) =====
# 실제로는 더 정교한 토크나이저 추천 (예: KoNLPy, Mecab, etc.)
def simple_tokenize(text):
    # 공백 단위로 토큰화, 필요시 문장부호 제거 등 추가 가능
    return text.lower().replace('?', '').replace('.', '').split()

# 단어 집합 구축
vocab = {"<PAD>": 0, "<UNK>": 1}
for item in data:
    tokens = simple_tokenize(item["text"])
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)

# ===== 4. Dataset 클래스 작성 =====
class IntentDataset(Dataset):
    def __init__(self, data, vocab, label_to_idx, max_len=5):
        self.data = data
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = simple_tokenize(item['text'])

        # max_len으로 패딩/자르기
        if len(tokens) < self.max_len:
            tokens += ["<PAD>"] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        # 토큰 -> 인덱스
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        # 멀티라벨 인코딩 (multi-hot 벡터)
        label_vec = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
        for label in item['labels']:
            label_idx = self.label_to_idx[label]
            label_vec[label_idx] = 1.0

        return torch.tensor(token_ids), label_vec

# ===== 5. 모델 정의 (앞서 만든 모델 사용) =====
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, max_len=5, num_classes=len(label_list)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.input_dim = embedding_dim * max_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, max_len, emb_dim)
        x = x.view(x.size(0), -1)  # flatten (batch, max_len*emb_dim)
        # 만약 부족하면 패딩 (보통 max_len 고정이므로 생략 가능)
        if x.size(1) < self.input_dim:
            pad_len = self.input_dim - x.size(1)
            padding = torch.zeros(x.size(0), pad_len, device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ===== 6. 학습 함수 =====
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for token_ids, labels in dataloader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(token_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * token_ids.size(0)
    return total_loss / len(dataloader.dataset)

# ===== 7. 학습 실행 코드 =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = IntentDataset(data, vocab, label_to_idx, max_len=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = IntentClassifier(vocab_size=len(vocab), embedding_dim=300, max_len=5, num_classes=len(label_list))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss:.4f}")