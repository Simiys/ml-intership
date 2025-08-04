from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
from bs4 import BeautifulSoup
import re
import time
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app) 

# Загружаем модель и токенизатор при старте приложения
print("Загружаем NER модель...")
try:
    model = AutoModelForTokenClassification.from_pretrained("./ner_model/checkpoint-125")
    tokenizer = AutoTokenizer.from_pretrained("./ner_model/checkpoint-125")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    print("✅ NER модель загружена успешно")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    ner_pipeline = None

def extract_possible_titles(url):
    """Извлекает потенциальные названия продуктов с веб-страницы"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"[SKIP] {url}: Status code {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "lxml")
        candidates = []

        # Извлекаем h1 заголовки
        for tag in soup.find_all(['h1']):
            text = tag.get_text(strip=True)
            if text:
                candidates.append(text)

        # Извлекаем элементы с классами, содержащими "name", "title"
        for tag in soup.find_all(['p', 'div', 'span']):
            cls = tag.get("class")
            if cls and any(any(substr in c.lower() for substr in ["name", "title"]) for c in (cls if isinstance(cls, list) else [cls])):
                text = tag.get_text(strip=True)
                if text:
                    candidates.append(text)

        return candidates

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {url}: {e}")
        return []

def process_with_ner(texts):
    """Обрабатывает список текстов через NER модель и возвращает результаты"""
    if not ner_pipeline:
        return []
    
    results = []
    
    for text in texts:
        if not text or len(text.strip()) < 2:
            continue
            
        try:
            # Получаем предсказания от модели
            entities = ner_pipeline(text)
            
            # Ищем сущности с меткой PRODUCT
            for entity in entities:
                if entity.get('entity_group') == 'PRODUCT' or entity.get('entity') == 'PRODUCT':
                    confidence = entity.get('score', 0)
                    # Конвертируем в проценты и округляем
                    prob_percent = f"{int(confidence * 100)}%"
                    
                    # Добавляем весь исходный текст, а не только найденную сущность
                    results.append({
                        "text": text,
                        "prob": prob_percent
                    })
                    break  # Прерываем, так как уже нашли PRODUCT в этом тексте
                    
        except Exception as e:
            print(f"Ошибка обработки текста '{text}': {e}")
            continue
    
    return results

@app.route('/api/analyze', methods=['POST'])
def analyze_url():
    """Основная функция для обработки URL и извлечения названий мебели"""
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "URL не предоставлен"}), 400
        
        url = data['url']
        
        # Валидация URL
        if not url.startswith(('http://', 'https://')):
            return jsonify({"error": "Некорректный URL"}), 400
        
        print(f"Обрабатываем URL: {url}")
        
        # Извлекаем потенциальные названия
        titles = extract_possible_titles(url)
        
        if not titles:
            return jsonify({"results": [], "message": "Не удалось извлечь названия с данной страницы"})
        
        print(f"Найдено {len(titles)} потенциальных названий")
        
        # Удаляем дубликаты, сохраняя порядок
        unique_titles = []
        seen = set()
        for title in titles:
            if title not in seen:
                unique_titles.append(title)
                seen.add(title)
        
        print(f"После удаления дубликатов: {len(unique_titles)} названий")
        
        # Обрабатываем через NER модель
        results = process_with_ner(unique_titles)
        
        # Сортируем по вероятности (убираем % и конвертируем в int для сортировки)
        results.sort(key=lambda x: int(x['prob'].replace('%', '')), reverse=True)
        
        print(f"Найдено {len(results)} продуктов мебели")
        
        return jsonify({
            "results": results,
            "total_titles_found": len(unique_titles),
            "products_identified": len(results)
        })
        
    except Exception as e:
        print(f"Ошибка при обработке запроса: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервиса"""
    model_status = "loaded" if ner_pipeline else "not_loaded"
    return jsonify({
        "status": "ok",
        "model_status": model_status
    })
    
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    file_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')    

if __name__ == '__main__':
    print("🚀 Запускаем Flask сервер...")
    print("📝 Доступные эндпоинты:")
    print("  POST /analyze - анализ URL на наличие названий мебели")
    print("  GET /health - проверка состояния сервиса")
    
    app.run(debug=True, host='0.0.0.0', port=5000)