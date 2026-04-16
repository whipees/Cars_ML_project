# Dokumentace softwarového projektu: Car ML Predictor

## 1. Úvodní informace
* **Název projektu:** AI Car Assistant (Car ML Predictor)
* **Autor:** Sebastian Janíček
* **Kontaktní údaje:** janicek@spsejecna.cz
* **Datum vypracování:** Duben 2026
* **Škola:** SPŠE Ječná
* **Upozornění:** Tento software byl vytvořen výhradně jako školní projekt.

---

## 2. Specifikace požadavků (Requirements)
Aplikace slouží jako inteligentní asistent pro autobazary a prodejce vozidel. 

**Functional Requirements (Funkční požadavky):**
* **UC1 (Rozpoznání značky):** Uživatel nahraje fotografii vozidla (.jpg, .png). Systém pomocí konvoluční neuronové sítě detekuje značku vozidla a zobrazí ji s procentuální jistotou.
* **UC2 (Predikce ceny):** Uživatel zadá parametry vozidla (Značka, Model, Rok, Nájezd, Motor, Palivo, Výkon, Pohon, Převodovka). Systém na základě modelu náhodného lesa (Random Forest) odhadne aktuální tržní cenu v CZK.
* **UC3 (Chytré filtrování):** Aplikace automaticky omezuje výběr v rozbalovacích seznamech (dropdown) na základě předchozí volby (např. po výběru značky "Škoda" nabídne pouze modely od Škody).

---

## 3. Architektura aplikace
Aplikace je navržena jako monolitická desktopová aplikace s oddělenou prezentační a logickou vrstvou.
* **Prezentační vrstva (UI):** Vytvořena v nativní Python knihovně `tkinter`. Stará se o vykreslení karet, tlačítek a zpracování událostí (event handling).
* **Datová vrstva:** Zajišťuje načítání souborů modelů (`.pkl`, `.h5`), encodérů pro kategorická data a `.csv` souboru s čistými daty pro kaskádové menu.
* **ML Vrstva (Integrovaná):** Využívá `TensorFlow/Keras` pro počítačové vidění (obraz) a `scikit-learn` pro regresní analýzu (cena).
* **Zpracování dat (Data Prep):** Oddělené skripty v adresáři `/src/data_prep/`, které neslouží pro běh aplikace, ale pro přípravu a čištění datasetů před trénováním.

---

## 4. Popis běhu aplikace (Behavior)
Běh aplikace (Activity flow) v typickém scénáři:
1. **Inicializace:** Uživatel spustí `run.bat`. Dávkový soubor ověří závislosti, aktivuje `.venv` a spustí `app.py`.
2. **Načtení modelů (Load State):** Aplikace načte `price_model.pkl` a `brand_classifier.h5`. Zablokuje výstup TensorFlow warningů.
3. **Čekání na vstup (Idle State):** Aplikace zobrazí GUI a čeká na interakci uživatele.
4. **Zpracování obrazu:** Uživatel vybere fotku -> Aplikace fotku zmenší na 224x224 px -> Převede na matici (Numpy array) -> Aplikuje MobileNetV2 preprocessing -> Provede predikci -> Vypíše výsledek a automaticky předvyplní značku do formuláře pro cenu.
5. **Zpracování ceny:** Uživatel vyplní zbytek formuláře -> Aplikace provede validaci a konverzi textu na čísla pomocí `LabelEncoder` -> Sestaví Pandas DataFrame ve striktním pořadí -> Provede predikci -> Zobrazí formátovanou cenu.

---

## 5. Rozhraní, knihovny třetích stran a specifikace
Aplikace nevyužívá externí API, veškeré výpočty probíhají lokálně (Offline AI/ML inference). 
**Klíčové závislosti (Non-functional requirements):**
* `Python 3.9+` (Základní běhové prostředí)
* `TensorFlow >= 2.16.1` (Inference konvoluční sítě MobileNetV2)
* `scikit-learn == 1.6.1` (Inference Random Forest a Label Encoders)
* `pandas`, `numpy` (Manipulace a transformace dat, maticové operace)
* `Pillow` (Práce s obrazovými soubory v GUI)
* `joblib` (Deserializace uložených strojových modelů)

---

## 6. Právní a licenční aspekty
* **Zdrojový kód aplikace:** Autorský kód vytvořený autorem projektu.
* **Datasety:** Data a fotografie pro trénování modelů byla získána autorem sběrem z veřejně dostupných zdrojů. Slouží výhradně pro edukativní účely (Fair Use) a nejsou komerčně distribuována.
* **Knihovny:** Všechny použité knihovny (TensorFlow, scikit-learn, atd.) podléhají svobodným open-source licencím (např. Apache 2.0, BSD, MIT).

---

## 7. Konfigurace
Aplikace je navržena jako "Zero-Config" pro koncového uživatele.
* **Rozšíření dat:** Pro přidání nových aut do rozbalovacích nabídek stačí updatovat soubor `cleaned_cars_data.csv` ve složce `/data/processed/`.

---

## 8. Instalace a spuštění
Aplikace nevyžaduje ruční instalaci knihoven. Byla navržena pro snadné spuštění (Deployment) v učebně:
1. Rozbalte složku s projektem.
2. Přejděte do složky `/bin`.
3. Spusťte soubor `run.bat`.
*Poznámka: Skript při prvním spuštění automaticky vytvoří virtuální prostředí (`.venv`) a stáhne všechny knihovny ze souboru `requirements.txt` umístěného v kořenovém adresáři.*

---

## 9. Chybové stavy a jejich řešení
Aplikace je plně ošetřena proti pádům pomocí rozsáhlých `try-except` bloků.
* **Chybějící Python (ERRORLEVEL):** `run.bat` detekuje absenci Pythonu a vyzve uživatele k instalaci (s nutností zaškrtnout ADD TO PATH).
* **Chyba načtení modelů (Deserialization error):** Ošetřeno přes třídu `SafeDense`, která odstraňuje nekompatibilní parametry (`quantization_config`) mezi různými verzemi TensorFlow.
* **Chybný formát vstupu u ceny:** Pokud uživatel zadá text tam, kde má být číslo, aplikace zachytí výjimku, vloží defaultní hodnotu `0` a zobrazí chybovou hlášku "Error in inputs", aniž by aplikace spadla.
* **Chybějící CSV pro filtrování:** Pokud chybí data, aplikace plynule přejde na čtení přímo ze slovníku Label Encodérů (Fallback mechanism).

---

## 10. Testování, ověření a validace
**A. Strojové učení (Machine Learning Metrics):**
Testování proběhlo odštěpením 20 % dat z vlastního datasetu (test_size=0.2, random_state=42), která model nikdy neviděl.
* **Decision Tree (Hloubka 3):** Sloužil pro analýzu rozhodovacích hranic a vizualizaci.
* **Random Forest (Produkční):** Dosahuje vysokého skóre predikce ceny na základě MSE a MAE metrik (Hodnoty viz výstup z Google Colab).
* **MobileNetV2 (Vision):** Testováno po 10 epochách s aplikací vrstvy "Data Augmentation" (RandomFlip, RandomRotation), která zabránila over-fittingu.

**B. Uživatelské testování (UI Validation):**
* Testováno nesmyslné zadávání znaků do číselných polí (ošetřeno).
* Testováno zadání fotografií neobsahujících auto (model se pokusí najít nejbližší rysy).
* Aplikace splňuje stanovené školní požadavky (ovladatelnost bez IDE).

---



## 11. Hardware, Databáze a Sítě
* **Databáze:** Aplikace nevyužívá relační databázi (SQL). Data nezbytná pro kaskádové filtrování a kódování textu jsou dodávána v serializované podobě (`.pkl`) nebo ve statických textových souborech (`.csv`, `.json`).
* **Sítě:** Aplikace ke svému běhu nevyžaduje připojení k síti Internet. Nevyužívá API třetích stran. Trénovací modely jsou obsaženy lokálně ve složce `/models/`.
* **Hardware:** Zpracování neuronové sítě probíhá na CPU počítače (s využitím instrukcí AVX/AVX2, pokud jsou k dispozici). GPU není pro inferenci tohoto modelu vyžadována.

---

## 122. Import a Export (Práce se soubory)
Aplikace provádí import lokálních souborů ze zařízení uživatele v rámci zpracování obrazu:
* **Pravidla importu:** Přes nativní dialog `tkinter filedialog` uživatel vybírá vizuální soubory. Aplikace podporuje standardní bitmapové a komprimované formáty (`.jpg`, `.jpeg`, `.png`).
* Soubory se do systému fyzicky nekopírují, dochází pouze k přečtení pixelové mapy a její transformaci do RGB tensoru 224x224 pro potřeby TensorFlow.