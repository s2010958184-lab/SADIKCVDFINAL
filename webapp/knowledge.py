"""Curated knowledge base of exercises, foods, and clinical tips.

Each entry is a dict the app uses for three things:
  1. Rendering animated cards on the results page (filtered by risk tier or flag).
  2. Acting as the document corpus for TF-IDF retrieval in retriever.py.
  3. Forming the grounded context block sent to Ollama in ollama_client.py.

All clinical content is drawn from the same sources the notebook cites:
WHO 2020 PA Guidelines, ACC/AHA 2023, ACSM FITT-VP, AHA Dietary Guidance 2021,
JNC 8, NCEP ATP III, ADA, DASH and Mediterranean diet evidence base.
"""
from __future__ import annotations

from typing import Any

# Tiers used by every entry
TIERS = ("Low Risk", "Moderate Risk", "High Risk", "Very High Risk")


# ---------------------------------------------------------------------------
# EXERCISE plans by risk tier (used as cards + retrieval docs)
# ---------------------------------------------------------------------------
EXERCISE_PLANS: dict[str, dict[str, Any]] = {
    "Low Risk": {
        "frequency": "5–7 days/week",
        "duration": "45–60 min/session",
        "intensity": "Moderate-to-vigorous (60–85% max HR)",
        "summary": "Build cardiovascular fitness with varied aerobic and resistance work.",
        "exercises": [
            {"emoji": "🏃", "name": "Brisk walking / jogging",
             "detail": "30–45 min at a pace where speaking is slightly challenging. Ideal baseline for cardiac fitness and BP control."},
            {"emoji": "🚴", "name": "Cycling",
             "detail": "40–60 min, adjust resistance progressively. Low joint impact, sustainable long-term."},
            {"emoji": "🏊", "name": "Swimming / aqua aerobics",
             "detail": "30–45 min. Full-body aerobic workout, minimal joint stress, excellent for BMI reduction."},
            {"emoji": "🏋️", "name": "Resistance training",
             "detail": "3×/week, 2–4 sets × 10–15 reps, all major muscle groups. Lowers resting HR, improves glucose metabolism."},
            {"emoji": "⚡", "name": "HIIT",
             "detail": "20–30 min: 40 s vigorous / 20 s rest. Highly effective for VO₂ max and insulin sensitivity."},
            {"emoji": "🧘", "name": "Yoga / Pilates",
             "detail": "2–3×/week, 30 min. Reduces cortisol — a CVD risk driver when chronically elevated."},
        ],
        "weekly_plan": [
            ("Monday", "Jogging 35 min + core 15 min"),
            ("Tuesday", "Cycling 45 min"),
            ("Wednesday", "Resistance training — full body 45 min"),
            ("Thursday", "Swimming 35 min"),
            ("Friday", "HIIT 25 min"),
            ("Saturday", "Yoga / stretching 40 min"),
            ("Sunday", "Rest or gentle walk 20 min"),
        ],
        "precautions": [
            "Warm up 5–10 min before every session.",
            "Hydrate: 500 ml before, 200 ml every 15–20 min during.",
            "Annual resting BP and cholesterol check from age 40.",
            "Stop if you feel chest tightness, dizziness, or unusual breathlessness.",
        ],
    },
    "Moderate Risk": {
        "frequency": "4–5 days/week",
        "duration": "30–45 min/session",
        "intensity": "Moderate (50–70% max HR) — able to hold a conversation",
        "summary": "Consistency beats intensity. Aerobic exercise is the single most evidence-backed intervention at this level.",
        "exercises": [
            {"emoji": "🚶", "name": "Brisk walking",
             "detail": "30–40 min, 5×/week. Target 8,000–10,000 steps/day."},
            {"emoji": "🚲", "name": "Stationary cycling",
             "detail": "30–40 min at low-to-moderate resistance. Controlled environment reduces risk vs outdoor."},
            {"emoji": "🏊", "name": "Swimming",
             "detail": "25–35 min. Hydrostatic pressure reduces cardiac preload — beneficial for elevated BP."},
            {"emoji": "🥾", "name": "Walk-jog intervals",
             "detail": "Alternate 2 min walk / 1 min jog for 30 min. Shift the ratio over 4–6 weeks."},
            {"emoji": "🧘", "name": "Yoga / Tai Chi",
             "detail": "2×/week, 30 min. Proven to reduce systolic BP by 5–10 mmHg over 8–12 weeks."},
            {"emoji": "🎽", "name": "Light resistance training",
             "detail": "2–3×/week, lighter loads. Never hold your breath — it spikes BP acutely."},
        ],
        "weekly_plan": [
            ("Monday", "Brisk walk 35 min"),
            ("Tuesday", "Resistance bands full body 30 min"),
            ("Wednesday", "Stationary cycling 35 min"),
            ("Thursday", "Rest or gentle yoga 30 min"),
            ("Friday", "Swimming 30 min"),
            ("Saturday", "Walk-jog intervals 30 min"),
            ("Sunday", "Full rest + stretching 15 min"),
        ],
        "precautions": [
            "Consult your GP before starting any new exercise programme.",
            "Resting HR should be under 100 bpm before moderate sessions.",
            "Check resting BP before exercise — postpone if > 160/100 mmHg.",
            "Avoid exercise in extreme heat or cold.",
            "Stop on chest pain, dizziness, palpitations, or severe breathlessness.",
        ],
    },
    "High Risk": {
        "frequency": "3–4 days/week",
        "duration": "20–30 min/session (build gradually)",
        "intensity": "Light-to-moderate (40–60% max HR) — speak comfortably throughout",
        "summary": "Exercise is still strongly recommended — but structured, supervised, and at controlled intensity. Start slow.",
        "exercises": [
            {"emoji": "🚶", "name": "Supervised walking",
             "detail": "20–30 min on flat terrain with an HR monitor. Safest starting point."},
            {"emoji": "💧", "name": "Water walking / aqua aerobics",
             "detail": "20–30 min. Water buoyancy reduces joint and cardiac load."},
            {"emoji": "🚲", "name": "Stationary bike — low resistance",
             "detail": "15–25 min, seated. Controlled environment, easy exit if symptoms arise."},
            {"emoji": "🪑", "name": "Chair-based exercises",
             "detail": "Seated leg raises, arm circles, seated marching. Effective for deconditioned patients."},
            {"emoji": "🌬️", "name": "Diaphragmatic breathing",
             "detail": "10 min daily. Lowers resting HR and BP when practised consistently."},
            {"emoji": "🌀", "name": "Tai Chi",
             "detail": "3×/week, 20 min. Evidence-based benefit in cardiac rehab populations."},
        ],
        "weekly_plan": [
            ("Monday", "Supervised walk 25 min (flat terrain only)"),
            ("Tuesday", "Rest + diaphragmatic breathing 10 min"),
            ("Wednesday", "Aqua aerobics 25 min"),
            ("Thursday", "Rest + gentle stretching 15 min"),
            ("Friday", "Stationary bike 20 min (low resistance)"),
            ("Saturday", "Chair yoga 20 min"),
            ("Sunday", "Rest — short gentle walk only if feeling well"),
        ],
        "precautions": [
            "PHYSICIAN CLEARANCE REQUIRED before starting any programme.",
            "Never exercise alone — have a companion present.",
            "Stop on chest pain, arm/jaw pain, severe breathlessness, or fainting.",
            "Carry GTN spray during all sessions if prescribed.",
            "Avoid isometric exercises (heavy planks, max lifts) — they spike BP.",
            "Target resting BP < 160/100 mmHg before each session.",
        ],
    },
    "Very High Risk": {
        "frequency": "3 supervised sessions/week + daily gentle activity",
        "duration": "15–25 min/session",
        "intensity": "Very light (30–50% max HR) — speak in full sentences at all times",
        "summary": "Formal Cardiac Rehabilitation is required before any independent exercise.",
        "exercises": [
            {"emoji": "🏥", "name": "Cardiac rehabilitation programme",
             "detail": "Phase II CR with telemetry monitoring, 3×/week for 12 weeks. Gold-standard programme."},
            {"emoji": "🚶", "name": "Slow walking — flat ground only",
             "detail": "10–15 min twice daily if approved by cardiologist."},
            {"emoji": "🪑", "name": "Seated aerobics",
             "detail": "15–20 min of gentle rhythmic seated movement."},
            {"emoji": "🌬️", "name": "Breathing & relaxation",
             "detail": "15 min daily. Reduces sympathetic activation."},
            {"emoji": "🛏️", "name": "Bed / chair stretches",
             "detail": "10 min morning routine. Improves circulation safely."},
        ],
        "weekly_plan": [
            ("Monday", "Cardiac rehab session (supervised) 25 min"),
            ("Tuesday", "Rest + breathing 15 min + gentle stretch"),
            ("Wednesday", "Cardiac rehab session (supervised) 25 min"),
            ("Thursday", "Rest + slow walk 10 min if medically approved"),
            ("Friday", "Cardiac rehab session (supervised) 25 min"),
            ("Saturday", "Gentle seated aerobics 15 min OR rest"),
            ("Sunday", "Full rest — focus on nutrition, hydration, sleep"),
        ],
        "precautions": [
            "MANDATORY: formal cardiology evaluation before any exercise.",
            "Enrol in Phase II Cardiac Rehabilitation if eligible.",
            "Exercise ONLY under medical supervision initially.",
            "Know your personalised target HR range from your cardiologist.",
            "All prescribed medications must be taken before exercise.",
            "Any new or changed symptoms = STOP and contact your medical team.",
        ],
    },
}


# ---------------------------------------------------------------------------
# DIET plans by risk tier
# ---------------------------------------------------------------------------
DIET_PLANS: dict[str, dict[str, Any]] = {
    "Low Risk": {
        "framework": "Mediterranean diet",
        "summary": "Heart-healthy maintenance pattern. Emphasise plants, fish, olive oil; minimise ultra-processed food.",
        "foods": [
            {"emoji": "🥗", "name": "Vegetables (5+ portions/day)",
             "detail": "Leafy greens, peppers, tomatoes, broccoli. Potassium counters sodium's BP effect."},
            {"emoji": "🍎", "name": "Fruits (2–3 portions/day)",
             "detail": "Berries, citrus, apples, bananas. Provide fibre, vitamin C, polyphenols."},
            {"emoji": "🐟", "name": "Oily fish (2×/week)",
             "detail": "Salmon, sardines, mackerel. Omega-3 lowers triglycerides — 36% reduction in cardiac death risk (AHA)."},
            {"emoji": "🥜", "name": "Nuts & seeds (a small handful/day)",
             "detail": "Walnuts, almonds. Replace saturated-fat snacks. ~30% CVD reduction with daily intake."},
            {"emoji": "🫒", "name": "Extra-virgin olive oil",
             "detail": "Primary cooking fat. 4 tbsp/day in the PREDIMED study reduced CVD events by 30%."},
            {"emoji": "🌾", "name": "Whole grains",
             "detail": "Oats, brown rice, whole-wheat bread, quinoa. Soluble fibre lowers LDL cholesterol."},
            {"emoji": "🫘", "name": "Legumes (3+ servings/week)",
             "detail": "Lentils, chickpeas, beans. High protein + fibre, very low saturated fat."},
        ],
        "avoid": [
            "Trans fats (industrial baked goods, some margarines).",
            "Sugary drinks > 1 can/week.",
            "Processed red meat > 2 servings/week.",
        ],
        "meals": [
            {"meal": "Breakfast", "emoji": "🥣", "name": "Greek yoghurt bowl",
             "detail": "Plain Greek yoghurt, berries, walnuts, a drizzle of honey. Calcium + omega-3 + antioxidants in one go."},
            {"meal": "Lunch", "emoji": "🥙", "name": "Mediterranean grain bowl",
             "detail": "Quinoa, chickpeas, cucumber, tomato, feta, olives, olive oil. ~600 kcal of fibre and unsaturated fat."},
            {"meal": "Dinner", "emoji": "🐟", "name": "Grilled salmon + roast veg",
             "detail": "150 g salmon, roasted broccoli, sweet potato, lemon. Omega-3 and beta-carotene."},
            {"meal": "Snack", "emoji": "🥜", "name": "Mixed nuts (30 g)",
             "detail": "Almonds, walnuts, brazil nuts. Replace crisp/biscuit snacking."},
        ],
    },
    "Moderate Risk": {
        "framework": "Mediterranean + light DASH overlay",
        "summary": "Tighten sodium and saturated-fat intake; expand plant proteins; oily fish twice weekly is non-negotiable.",
        "foods": [
            {"emoji": "🥬", "name": "Leafy greens daily",
             "detail": "1–2 cups raw or ½ cup cooked daily. Rich in nitrates, potassium and folate."},
            {"emoji": "🐟", "name": "Oily fish 2–3×/week",
             "detail": "Salmon, sardines, mackerel. Omega-3 reduces triglycerides and inflammation."},
            {"emoji": "🥑", "name": "Avocado, nuts, olive oil",
             "detail": "Replace butter and cream. Monounsaturated fat improves the LDL:HDL ratio."},
            {"emoji": "🍓", "name": "Berries (5×/week)",
             "detail": "Anthocyanins lower BP. Strong observational evidence."},
            {"emoji": "🌾", "name": "Whole-grain switch",
             "detail": "Brown rice, whole oats, whole-wheat pasta. Avoid refined white-flour staples."},
            {"emoji": "🧄", "name": "Garlic + herbs replace salt",
             "detail": "Garlic, basil, oregano, lemon. Cuts sodium load while keeping flavour."},
            {"emoji": "🫛", "name": "Plant proteins 4×/week",
             "detail": "Lentils, beans, tofu, chickpeas in place of red meat."},
        ],
        "avoid": [
            "Sodium > 2,000 mg/day (1 tsp of salt).",
            "Sugary drinks completely.",
            "Processed meats (bacon, salami, sausages).",
            "Deep-fried foods.",
        ],
        "meals": [
            {"meal": "Breakfast", "emoji": "🥣", "name": "Steel-cut oats with berries",
             "detail": "½ cup oats, blueberries, a tbsp ground flaxseed, cinnamon. Soluble fibre lowers LDL by 5–10%."},
            {"meal": "Lunch", "emoji": "🥗", "name": "Big leafy salad with sardines",
             "detail": "Spinach, tomato, avocado, tinned sardines in olive oil, lemon juice. No salt added."},
            {"meal": "Dinner", "emoji": "🍲", "name": "Lentil + vegetable stew",
             "detail": "Red lentils, carrots, courgette, garlic, herbs. High potassium for BP, plant protein swap for red meat."},
            {"meal": "Snack 1", "emoji": "🍎", "name": "Apple + 30 g walnuts",
             "detail": "Pectin fibre + omega-3. Replaces afternoon biscuits."},
            {"meal": "Snack 2", "emoji": "🥒", "name": "Hummus + raw veg",
             "detail": "2 tbsp hummus, carrot and cucumber sticks. No-salt brand only."},
        ],
    },
    "High Risk": {
        "framework": "DASH (Dietary Approaches to Stop Hypertension)",
        "summary": "Aggressively low sodium, low saturated fat, very high potassium. Hard limit on alcohol.",
        "foods": [
            {"emoji": "🥬", "name": "Vegetables 4–5 servings/day",
             "detail": "DASH core. Potassium-rich greens, peppers, root vegetables."},
            {"emoji": "🍎", "name": "Fruits 4–5 servings/day",
             "detail": "Bananas, oranges, melons, berries. Highest potassium for BP lowering."},
            {"emoji": "🥛", "name": "Low-fat dairy 2–3 servings/day",
             "detail": "Skimmed milk, low-fat yoghurt. Calcium + potassium combination."},
            {"emoji": "🐟", "name": "Lean fish or poultry",
             "detail": "Replace red meat. White fish, chicken breast, turkey."},
            {"emoji": "🥜", "name": "Nuts/seeds/legumes 4–5×/week",
             "detail": "Magnesium-rich. Lower BP further when combined with vegetable intake."},
            {"emoji": "🥣", "name": "Oats every morning",
             "detail": "Beta-glucan soluble fibre lowers LDL by 5–10% over 8 weeks."},
            {"emoji": "🌶️", "name": "Spices for flavour (no salt)",
             "detail": "Black pepper, paprika, cinnamon, turmeric replace sodium."},
        ],
        "avoid": [
            "Sodium > 1,500 mg/day (~⅔ tsp salt).",
            "All processed meats.",
            "Alcohol > 1 drink/day (women) or 2 drinks/day (men) — ideally minimal.",
            "Ultra-processed snacks and sauces.",
            "Coconut and palm oil.",
        ],
        "meals": [
            {"meal": "Breakfast", "emoji": "🥣", "name": "DASH-style porridge",
             "detail": "½ cup steel-cut oats, skimmed milk, sliced banana, walnuts, cinnamon. 4 g fibre, no added salt."},
            {"meal": "Lunch", "emoji": "🥗", "name": "Skinless chicken + giant salad",
             "detail": "100 g grilled chicken breast, 4 cups mixed greens, beets, beans, oil + vinegar dressing."},
            {"meal": "Dinner", "emoji": "🐟", "name": "White fish + steamed veg",
             "detail": "Cod or haddock, broccoli, carrots, brown rice. Herbs and lemon — zero salt."},
            {"meal": "Snack 1", "emoji": "🍌", "name": "Banana + 200 ml skimmed milk",
             "detail": "Potassium + magnesium combination targets BP directly."},
            {"meal": "Snack 2", "emoji": "🍇", "name": "Berry bowl",
             "detail": "Mixed berries (200 g). Anthocyanins lower BP."},
            {"meal": "Hydration", "emoji": "💧", "name": "1.5–2 L water/day",
             "detail": "Plain water or unsweetened herbal tea. Avoid sodium-laden sports drinks."},
        ],
    },
    "Very High Risk": {
        "framework": "Strict DASH + clinician supervision",
        "summary": "Diet is medically supervised — should be aligned with prescribed medications. Sodium < 1,500 mg/day mandatory.",
        "foods": [
            {"emoji": "🥦", "name": "DASH-eating-plan portions",
             "detail": "9–11 servings of fruit + vegetables a day, prescribed by dietitian."},
            {"emoji": "💧", "name": "Hydration: 1.5–2 L/day water",
             "detail": "Unless your cardiologist restricts fluid intake."},
            {"emoji": "🐟", "name": "Oily fish 2–3×/week (or omega-3 supplement)",
             "detail": "Consult cardiologist re. interaction with anticoagulants."},
            {"emoji": "🍠", "name": "Low-glycaemic carbs only",
             "detail": "Sweet potato, lentils, steel-cut oats. Stable blood sugar reduces cardiac stress."},
            {"emoji": "🥚", "name": "Lean protein, small portions",
             "detail": "Egg whites, white fish, tofu. ≤ 90 g cooked per meal."},
            {"emoji": "🌿", "name": "Herbal seasoning only",
             "detail": "Salt is a strict no. Read every label."},
        ],
        "avoid": [
            "All added salt and high-sodium processed foods (sodium < 1,500 mg/day).",
            "Alcohol — discuss any intake with your cardiologist.",
            "Caffeine in excess (> 2 cups coffee/day).",
            "Grapefruit if on statins (drug interaction).",
            "Trans fats in any form.",
        ],
        "meals": [
            {"meal": "Breakfast", "emoji": "🥣", "name": "Oats with seeds (small portion)",
             "detail": "⅓ cup steel-cut oats, chia seeds, sliced strawberries. Low sodium, high fibre, easy on digestion."},
            {"meal": "Lunch", "emoji": "🥗", "name": "Small white-fish portion + steamed veg",
             "detail": "80 g cod, steamed green beans, ½ cup brown rice. Herbs only — strictly no salt."},
            {"meal": "Dinner", "emoji": "🍵", "name": "Vegetable soup (no salt)",
             "detail": "Carrot, celery, courgette, garlic, low-sodium stock. Soft, gentle, hydrating."},
            {"meal": "Snack 1", "emoji": "🍌", "name": "Banana",
             "detail": "Potassium-rich, simple, easy on the GI tract."},
            {"meal": "Snack 2", "emoji": "🍎", "name": "Apple slices + cinnamon",
             "detail": "Skin-on apple. Cinnamon improves insulin sensitivity."},
            {"meal": "Hydration", "emoji": "💧", "name": "1.5 L water/day (check with cardiologist)",
             "detail": "Some heart-failure patients need fluid restriction — confirm your daily allowance."},
        ],
    },
}


# ---------------------------------------------------------------------------
# Daily tips — used both as cards and as RAG documents
# ---------------------------------------------------------------------------
DAILY_TIPS: list[dict[str, Any]] = [
    {"category": "Nutrition", "tier": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "Eat 5 portions of fruits and vegetables every day.",
     "why": "Each extra daily serving cuts CVD mortality by ~4–5% (16-study meta-analysis).",
     "action": "Add a banana to breakfast and two veg to lunch — already 3 portions.",
     "source": "WHO Global Action Plan for NCDs"},
    {"category": "Nutrition", "tier": ["Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "Keep sodium under 2,000 mg (2 g) a day.",
     "why": "Every 1 g/day reduction lowers systolic BP by ~2 mmHg.",
     "action": "Avoid salt at the table. Scan labels — one can of soup can exceed 900 mg.",
     "source": "WHO: Sodium Intake for Adults"},
    {"category": "Nutrition", "tier": ["Low Risk", "Moderate Risk", "High Risk"],
     "tip": "Eat oily fish twice a week for omega-3 protection.",
     "why": "EPA and DHA reduce triglycerides and platelet clumping. 2 servings/week reduces cardiac death risk by 36%.",
     "action": "Salmon on Tuesday, sardines on Friday.",
     "source": "AHA Fish and Omega-3 Recommendation"},
    {"category": "Physical Activity", "tier": ["Low Risk", "Moderate Risk", "High Risk"],
     "tip": "150 minutes of moderate aerobic activity per week is the single most evidence-backed CVD prevention.",
     "why": "150 min/week reduces CVD risk by 35% and all-cause mortality by 33%.",
     "action": "30 minutes, 5 days a week. A brisk walk counts.",
     "source": "WHO Global Recommendations on Physical Activity"},
    {"category": "Physical Activity", "tier": ["Low Risk", "Moderate Risk"],
     "tip": "Break up sitting time every 30 minutes.",
     "why": "Prolonged sitting raises CVD risk markers independently of total exercise.",
     "action": "Stand up and walk 2 minutes every half hour at your desk.",
     "source": "BJSM sedentary behaviour reviews"},
    {"category": "Blood Pressure", "tier": ["Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "Measure BP at home twice a week.",
     "why": "Home BP correlates better with cardiovascular outcomes than clinic BP alone.",
     "action": "Same time of day, seated 5 min, arm at heart level. Log it.",
     "source": "ESH 2023 BP Monitoring Guidelines"},
    {"category": "Sleep & Stress", "tier": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "Sleep 7–9 hours a night — both too little and too much raise CVD risk.",
     "why": "Short sleep (<6 h) raises CVD risk by ~20%; chronic poor sleep elevates resting BP and cortisol.",
     "action": "Fixed sleep window, no screens 30 min before bed.",
     "source": "AHA Sleep Duration and CVD Statement"},
    {"category": "Smoking & Alcohol", "tier": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "If you smoke, quitting is the single highest-impact intervention you can make.",
     "why": "CVD risk drops by ~50% within 1 year of quitting.",
     "action": "Set a quit date. Use NRT + behavioural support — combo wins.",
     "source": "ACC/AHA 2019 Primary Prevention Guideline"},
    {"category": "Cholesterol & Glucose", "tier": ["Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "Replace saturated fats with unsaturated fats.",
     "why": "Replacing 5% of saturated-fat calories with polyunsaturated fat reduces coronary disease risk by ~10%.",
     "action": "Olive oil instead of butter; nuts instead of crisps.",
     "source": "ACC/AHA 2019 Primary Prevention Guideline"},
    {"category": "Screening & Weight", "tier": ["Moderate Risk", "High Risk", "Very High Risk"],
     "tip": "A 5% weight loss lowers BP by ~5 mmHg and improves lipids materially.",
     "why": "Dose-response confirmed in randomised trials.",
     "action": "Target a slow loss of 0.5 kg per week — sustainable beats fast.",
     "source": "NIH/NHLBI Obesity Guidelines"},
]


# ---------------------------------------------------------------------------
# Emoji → animated-SVG icon mapping. Used by the Jinja {{ icon() }} macro in
# templates/_icons.html. Unknown emojis fall back to the original emoji glyph
# wrapped in a CSS bob animation, so adding new entries is forgiving.
# ---------------------------------------------------------------------------
EMOJI_TO_ICON: dict[str, str] = {
    # exercise
    "🏃": "running",
    "🚴": "cycling",
    "🚲": "cycling",
    "🏊": "swimming",
    "🏋️": "lifting",
    "🎽": "lifting",
    "💪": "lifting",
    "🚶": "walking",
    "🥾": "running",
    "🧘": "yoga",
    "🌀": "yoga",
    "🌬️": "yoga",
    "🪑": "yoga",
    "🛏️": "yoga",
    "⚡": "running",
    "🏥": "heart",
    "💧": "water",
    # diet
    "🥗": "salad",
    "🥬": "salad",
    "🥦": "salad",
    "🍎": "apple",
    "🍌": "apple",
    "🍠": "apple",
    "🐟": "fish",
    "🥜": "grain",
    "🫒": "olive-oil",
    "🌾": "grain",
    "🥣": "grain",
    "🫘": "beans",
    "🫛": "beans",
    "🥑": "avocado",
    "🍓": "berry",
    "🍇": "berry",
    "🥛": "milk",
    "🥚": "milk",
    "🌶️": "grain",
    "🌿": "grain",
    "🧄": "grain",
    "🥒": "salad",
    "🥙": "salad",
    "🍲": "salad",
    "🍵": "salad",
}


def icon_name(emoji: str) -> str | None:
    """Return the animated-SVG icon name registered for an emoji, or None."""
    return EMOJI_TO_ICON.get(emoji)


# ---------------------------------------------------------------------------
# Helpers used by app.py
# ---------------------------------------------------------------------------
def plan_for_tier(tier: str) -> dict[str, Any]:
    return {
        "exercise": EXERCISE_PLANS[tier],
        "diet": DIET_PLANS[tier],
        "tips": [t for t in DAILY_TIPS if tier in t["tier"]],
    }


def all_knowledge_documents() -> list[dict[str, Any]]:
    """Flattens the corpus into TF-IDF-ready documents for retriever.py."""
    docs: list[dict[str, Any]] = []
    for tier, plan in EXERCISE_PLANS.items():
        for it in plan["exercises"]:
            docs.append({
                "kind": "exercise",
                "tier": tier,
                "title": it["name"],
                "emoji": it["emoji"],
                "text": f"Exercise for {tier}: {it['name']}. {it['detail']} "
                        f"Frequency {plan['frequency']}, duration {plan['duration']}, "
                        f"intensity {plan['intensity']}.",
            })
    for tier, plan in DIET_PLANS.items():
        for it in plan["foods"]:
            docs.append({
                "kind": "diet",
                "tier": tier,
                "title": it["name"],
                "emoji": it["emoji"],
                "text": f"Diet for {tier} ({plan['framework']}): {it['name']}. {it['detail']}",
            })
    for t in DAILY_TIPS:
        docs.append({
            "kind": "tip",
            "tier": ",".join(t["tier"]),
            "title": t["tip"],
            "emoji": "💡",
            "text": f"Daily tip — {t['category']}. {t['tip']} Why: {t['why']} "
                    f"Action: {t['action']} Source: {t['source']}.",
        })
    return docs
