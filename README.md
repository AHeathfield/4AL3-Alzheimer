# 4AL3-Alzheimer

## How to Run
### Install requirements, venv
```bash
cd py/
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

### Preprocess the data
```bash
python preprocess.py
```
- This will produce the features.npz (numpy zipped)

### Train the data
```bash
python train.py
```
