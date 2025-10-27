# 🤟 Sign2Speech

A simple AI-based application that converts **sign language gestures** into **text and speech** using a webcam.  
It helps deaf and mute individuals communicate easily with people who don’t know sign language.

---

## 🧩 Overview
- Captures live hand gestures through a webcam  
- Predicts the gesture meaning using a trained ML model  
- Displays the recognised word as text  
- Speaks the text aloud using Text-to-Speech  

---

## 🗂 Project Files
| File | Description |
|------|--------------|
| `app.py` | Main app – runs webcam, detects sign, shows text + speech |
| `handsign.py` | Predicts gesture from webcam (no UI) |
| `train_model.py` | Trains the model using dataset (`sign_data.csv`) |
| `sign_model.pkl` | Saved trained model |
| `sign_data.csv` | Custom dataset (captured using webcam) |

---

## 🧠 Dataset Info
- **Source:** Custom images captured through webcam  
- **Classes (17):** Father, Fine, Finish, Food, Good bye, Hello, Help, Meeting, Mother, Name, No, Please, See you Later, Stop, Thank you, Yes, Sorry  
- **Samples:** ~3,400  
- **Stored in:** `sign_data.csv`

---

## ⚙️ How It Works
1. **Capture gesture** → Webcam records hand sign  
2. **Preprocess** → Frame resized and normalised  
3. **Predict** → Trained model (`sign_model.pkl`) identifies gesture  
4. **Display** → Shows recognised text  
5. **Speak** → Converts text to speech (pyttsx3)

---

## 🧮 Model Training
Run this to train:
```bash
python train_model.py
