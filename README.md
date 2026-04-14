# ArtificialIntelligence-UniversityProject

## Projekt: Prognozowanie ryzyka wystąpienia chorób przewlekłych
Wykorzystanie sieci neuronowych do analizy danych medycznych pacjentów (np. ciśnienie krwi, poziom cukru, wskaźnik BMI) w celu przewidywania ryzyka wystąpienia chorób takich jak cukrzyca, nadciśnienie czy chorby serca.

## Podsumowanie EDA

### Kluczowe wnioski:

1. **Niezbalansowane klasy** — ~65% zdrowych vs ~35% chorych -> potrzebne techniki balansowania (SMOTE, class weights)

2. **Brakujące wartości** zakodowane jako 0:
   - Insulin (~49%) i SkinThickness (~30%) - dużo braków
   - Glucose, BloodPressure, BMI - niewielkie braki
   
3. **Najsilniejsze predyktory cukrzycy:**
   - Glucose (stężenie glukozy) - najwyższa korelacja z Outcome
   - BMI (wskaźnik masy ciała)
   - Age (wiek)
   - Pregnancies (liczba ciąż)

4. **Outliery** - obecne w większości cech, szczególnie w Insulin i SkinThickness

5. **Korelacje między cechami** - umiarkowane, brak silnej multikolinearności