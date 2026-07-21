import os

replacements = {
    "وحدة التحكم": "كونسول",
    "الحاسب الآلي": "كمبيوتر",
    "المحرك": "محرك",
    "نظام التشغيل": "نظام تشغيل",
    "المجموعة": "مجموعة",
    "المجلد": "مجلد",
    "المحاكي": "محاكى",
    "محيطي": "طرفية"
}

current_dir = os.getcwd()
for filename in os.listdir(current_dir):
    if filename.endswith(".xml"):
        file_path = os.path.join(current_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        for old_word, new_word in replacements.items():
            content = content.replace(old_word, new_word)
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        
        print(f"The file has been updated: {filename}")