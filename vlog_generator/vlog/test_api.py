import requests

# 對應 main.py 設定的 2026 port
url = "http://127.0.0.1:2026/api/create_vlog" 

# 只需要填寫檔名，伺服器會自己去對應的資料夾找
payload = {
    "image_files": ["img5.jpg"],  
    "prompt": "Cinematic pan over a lush green tea garden, soft morning mist, highly detailed, 4k resolution, gentle breeze moving the leaves",
    "tts_audio_file": "07071049.wav", 
    "bgm_file": "dog_barking1.mp3"                 
}

print("🚀 發送 Vlog 生成請求給 API...")
response = requests.post(url, json=payload)
print(response.status_code)
print(response.text)

print("✅ 伺服器回應：") 
print(response.json())
