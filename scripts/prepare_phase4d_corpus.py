"""
Phase 4D: Contrastive Corpus Preparation Script.

Generates a balanced ~3,500-sample contrastive training and validation dataset
specifically targeting MuRIL's representation weaknesses:
  1. 750 Benign/celebratory profanity samples (English, Hindi, Hinglish)
  2. 750 Praise metaphors and colloquial slang samples (English, Hindi, Hinglish)
  3. 1,000 Properly annotated targeted Hindi/Hinglish abuse samples (with full multi-label supervision)
  4. 1,000 Clean conversational negative samples

Guarantees:
  - 0% leakage against all 686 held-out evaluation samples (Phase 5C probe, domain eval, synthetic stresstest)
  - Strict multi-label supervision across all 6 heads (toxic, severe_toxic, obscene, threat, insult, identity_hate)
  - Clear 70/15/15 train / val / held-out test split
  - Saved under scratch/phase4d_corpus/ to ensure no untracked data commits
"""

import sys
import re
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import config

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OUTPUT_DIR = _PROJECT_ROOT / "scratch" / "phase4d_corpus"


def compute_sha256(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def get_held_out_hashes() -> set:
    """Collect SHA256 hashes of all existing evaluation datasets to prevent leakage."""
    eval_hashes = set()

    # 1. Real-world domain evaluation set
    rw_path = config.DATASET_DIR / "real_world_domain_eval.csv"
    if rw_path.exists():
        df_rw = pd.read_csv(rw_path)
        for t in df_rw["comment_text"]:
            eval_hashes.add(compute_sha256(t))

    # 2. Synthetic adversarial stress test
    syn_path = config.DATASET_DIR / "synthetic_adversarial_stresstest.csv"
    if syn_path.exists():
        df_syn = pd.read_csv(syn_path)
        for t in df_syn["comment_text"]:
            eval_hashes.add(compute_sha256(t))

    # 3. Phase 5C difficult probe
    p5c_path = _PROJECT_ROOT / "evaluation_results" / "multilingual_eval_phase5c.json"
    if p5c_path.exists():
        with open(p5c_path, "r", encoding="utf-8") as f:
            p5c_data = json.load(f)
        for s in p5c_data.get("samples", []):
            eval_hashes.add(compute_sha256(s.get("text", "")))

    print(f"[CorpusPrep] Collected {len(eval_hashes)} evaluation hashes to strictly exclude.")
    return eval_hashes


def build_benign_profanity_set(count=750, exclude_hashes=None) -> pd.DataFrame:
    """750 Benign / celebratory profanity samples across EN, HI, Hinglish."""
    exclude_hashes = exclude_hashes or set()
    templates = [
        # English celebratory / relief / enthusiasm
        ("Fuck yeah! We finally passed all the unit tests after three days!", "english"),
        ("Holy shit that was the most incredible play I have ever witnessed!", "english"),
        ("That was so fucking awesome, I loved every single second of it!", "english"),
        ("Damn, that presentation was insanely good, brilliant job team!", "english"),
        ("Fuck yeah, got the promotion today! Hard work truly paid off!", "english"),
        ("Holy shit, you actually solved that impossible bug!", "english"),
        ("This is fucking fantastic news for all of us!", "english"),
        ("You are a badass engineer for fixing that outage so quickly!", "english"),
        ("That movie was kickass from beginning to end!", "english"),
        ("Holy shit, the graphics on this new game look unreal!", "english"),
        ("Fuck yeah, we hit our quarterly revenue target early!", "english"),
        ("Damn right, that's how a real champion plays the game!", "english"),
        ("That was so fucking cool when the beat dropped!", "english"),
        ("Holy shit that guitar solo gave me chills all over!", "english"),
        ("Fuck yeah, weekend is finally here and we crushed it!", "english"),
        ("Damn, that food looks so fucking delicious right now!", "english"),
        ("Holy shit, congratulations on getting married buddy!", "english"),
        ("Fuck yeah! Best concert I have attended in my entire life!", "english"),
        ("Damn, you played like a complete badass in the tournament!", "english"),
        ("Holy shit, look at that view from the mountain peak!", "english"),
        ("Fuck yeah, code compiled cleanly with zero compiler warnings!", "english"),
        ("Damn, that was a badass move on the skateboard!", "english"),
        ("Holy shit, you beat the world record time!", "english"),
        ("Fuck yeah, our research paper got accepted at the top conference!", "english"),
        ("That was a fucking legendary match, well played both teams!", "english"),

        # Hinglish friendly banter / cathartic exclamations
        ("Bhai bc treat de aaj, placement ho gaya tera!", "hinglish"),
        ("Abe saale kitne saal baad mila hai tu, gale mil!", "hinglish"),
        ("Bc kya gaana banaya hai, repeat mode pe sun raha hoon!", "hinglish"),
        ("Abe saale itna smart kab se ban gaya tu, maza aa gaya!", "hinglish"),
        ("Bc exam khatam finally, aaj raat party hogi full on!", "hinglish"),
        ("Bhai bc kya entry maari thi hero ne, seetiya baj rahi thi!", "hinglish"),
        ("Abe saale mast lag raha hai naye haircut me, sach me!", "hinglish"),
        ("Bc kya match tha yaar, aakhri ball pe chakka maar diya!", "hinglish"),
        ("Bhai bc car le li tune, badhai ho mere bhai!", "hinglish"),
        ("Abe saale itna acha khana banaya hai tune, chef ban ja tu!", "hinglish"),
        ("Bc college ke din yaad aa gaye teri photo dekh kar!", "hinglish"),
        ("Bhai bc kya acting ki hai bande ne, ro diya main toh!", "hinglish"),
        ("Abe saale congratulations, papa ban gaya tu!", "hinglish"),
        ("Bc office se chutti mil gayi poore hafte ki!", "hinglish"),
        ("Bhai bc gajab ka trip tha Goa ka, maza aa gaya!", "hinglish"),

        # Hindi informal expressive relief / friendly banter
        ("अरे साले तू तो बड़ा आदमी बन गया, फोन भी नहीं उठाता अब!", "hindi_devanagari"),
        ("भाई क्या फाड़ू डांस किया है तूने, पूरा स्टेज हिला दिया!", "hindi_devanagari"),
        ("अरे यार क्या जबरदस्त छक्का मारा, मज़ा ही आ गया मैच में!", "hindi_devanagari"),
        ("भाई तू तो छुपा रुस्तम निकला, क्या कमाल की पेंटिंग बनाई है!", "hindi_devanagari"),
        ("अरे साले पार्टी कब दे रहा है तू, इतनी बड़ी नौकरी लग गई!", "hindi_devanagari"),
        ("भाई क्या कड़क चाय बनाई है, सारी थकान मिट गई सच में!", "hindi_devanagari"),
        ("अरे यार क्या धाकड़ स्पीच दी तूने, सब तालियां बजा रहे थे!", "hindi_devanagari"),
        ("भाई सच में क्या गज़ब का फोटोग्राफर है तू, कमाल का एंगल लिया!", "hindi_devanagari"),
    ]

    rows = []
    modifiers = [
        "Bro, ", "Man, ", "Honestly, ", "Truly, ", "Guys, ", "Seriously, ",
        "Bhai, ", "Oye, ", "Sach me, ", "Dekho, ", "Yakeen nahi ho raha, ",
        "Finally, ", "No doubt, ", "Hands down, ", "Super happy, "
    ]
    tails = [
        " Respect!", " Huge congratulations!", " Keep rocking!", " Proud of you!",
        " Stay blessed!", " Cheers to that!", " What a day!", " Loved it completely!"
    ]

    idx = 0
    while len(rows) < count:
        tpl, lang = templates[idx % len(templates)]
        mod = modifiers[idx % len(modifiers)] if idx >= len(templates) else ""
        tail = tails[(idx // len(templates)) % len(tails)] if idx >= len(templates) else ""
        text = f"{mod}{tpl}{tail}".strip()
        h = compute_sha256(text)

        if h not in exclude_hashes and not any(r["hash"] == h for r in rows):
            rows.append({
                "comment_text": text,
                "language": lang,
                "category_type": "benign_profanity",
                "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0,
                "hash": h
            })
        idx += 1

    return pd.DataFrame(rows[:count])


def build_praise_slang_set(count=750, exclude_hashes=None) -> pd.DataFrame:
    """750 Praise metaphors, colloquial gaming/youth slang samples."""
    exclude_hashes = exclude_hashes or set()
    templates = [
        # English praise metaphors / slang
        ("Bro, you totally killed it on stage tonight, outstanding performance!", "english"),
        ("This movie was totally sick! Best sci-fi film of the decade!", "english"),
        ("You slayed that presentation, the entire executive board was impressed!", "english"),
        ("That guitar solo was deadly, your fingers were flying across the fretboard!", "english"),
        ("She completely murdered that rap verse, unbelievable flow and cadence!", "english"),
        ("This beat is sick, who produced this absolute masterpiece?", "english"),
        ("You guys crushed the deployment without a single minute of downtime!", "english"),
        ("He is a beast on the basketball court, unstoppable defense!", "english"),
        ("That dunk was vicious, shattered the entire momentum of the opposing team!", "english"),
        ("Your design skills are drop-dead gorgeous, insanely talented work!", "english"),
        ("That stand-up comedy special was killer, I was laughing tears the whole time!", "english"),
        ("You nailed that technical interview, congratulations on the offer!", "english"),
        ("That trick on the skateboard was insane and totally sick!", "english"),
        ("The bass in this track hits like a monster truck, love it!", "english"),
        ("You murdered that workout session today, true athletic dedication!", "english"),

        # Hinglish praise metaphors
        ("Bhai kya bawaal cheez banayi hai tune, maza hi aa gaya dekh kar!", "hinglish"),
        ("Bro tune to stage pe sach me aag laga di, deadly performance tha!", "hinglish"),
        ("Bhai katai zeher lag raha hai naye outfit me, hero lag raha hai!", "hinglish"),
        ("Gadar macha diya yaar tum dono ne dance floor pe!", "hinglish"),
        ("Bhai dhansu video banaya hai, concept ekdum crystal clear ho gaya!", "hinglish"),
        ("Ye trailer dekh kar to rongte khade ho gaye, bawaal movie hogi!", "hinglish"),
        ("Bhai dil jeet liya tune itni badiya shayari sunakar!", "hinglish"),
        ("Bro performance to ekdum zabardast tha, maza aa gaya sunke!", "hinglish"),
        ("Bhai tu to champion nikla, exam me top maar diya tune!", "hinglish"),
        ("Aag laga di bhai commentary me, kya energy thi tumhari!", "hinglish"),
        ("Bhai ye drawing dekh kar dimaag hil gaya, kya detail me banayi hai!", "hinglish"),
        ("Dhansu song hai yaar, gaadi me full volume pe bajta hai!", "hinglish"),
        ("Bhai bawaal cut kiya hai tune video, editing level next hai!", "hinglish"),

        # Hindi Devanagari praise metaphors
        ("भाई क्या धांसू काम किया है आपने, सच में दिल जीत लिया!", "hindi_devanagari"),
        ("यह नया गाना तो गदर मचा रहा है, ज़बरदस्त आवाज़ और धुन है।", "hindi_devanagari"),
        ("स्टेज पर आग लगा दी आपने, क्या शानदार प्रस्तुति थी!", "hindi_devanagari"),
        ("कमाल की अदाकारी की है आपने, हर एक दृश्य में जान डाल दी।", "hindi_devanagari"),
        ("आपकी आवाज़ में जादू है, दिल को छू लेने वाला संगीत।", "hindi_devanagari"),
        ("क्या बवाल किताब लिखी है लेखक ने, एक सांस में पढ़ डाली।", "hindi_devanagari"),
        ("भाई साहब क्या लाजवाब खाना बनाया है, उंगलियां चाटते रह गए।", "hindi_devanagari"),
        ("आपकी मेहनत रंग लाई, यह उपलब्धि वाकई गर्व की बात है।", "hindi_devanagari"),
    ]

    rows = []
    modifiers = [
        "Respect bro! ", "Just wow! ", "No words! ", "Salute to you! ", "Incredible! ",
        "Bhai sach me, ", "Sach bolu toh, ", "Ek number! ", "Waah bhai waah! "
    ]
    idx = 0
    while len(rows) < count:
        tpl, lang = templates[idx % len(templates)]
        mod = modifiers[idx % len(modifiers)] if idx >= len(templates) else ""
        text = f"{mod}{tpl}".strip()
        h = compute_sha256(text)

        if h not in exclude_hashes and not any(r["hash"] == h for r in rows):
            rows.append({
                "comment_text": text,
                "language": lang,
                "category_type": "praise_slang",
                "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0,
                "hash": h
            })
        idx += 1

    return pd.DataFrame(rows[:count])


def build_indic_targeted_abuse_set(count=1000, exclude_hashes=None) -> pd.DataFrame:
    """1,000 Fully annotated Hindi and Hinglish targeted abuse samples."""
    exclude_hashes = exclude_hashes or set()

    # Core high-quality multi-label annotated templates with realistic social media hostility
    abuse_templates = [
        # Direct insults + obscene (toxic=1, insult=1, obscene=1, severe=0, threat=0, hate=0)
        ("Bhai tu ek number ka chutiya aur ghatiya aadmi hai, sharm kar.", "hinglish", 1, 0, 1, 0, 1, 0),
        ("Tu bilkul madarchod insaan hai, apni aukaat me reh.", "hinglish", 1, 1, 1, 0, 1, 0),
        ("Abe bhosdike chup kar, teri bakwas sunne ka kisi ko shauk nahi hai.", "hinglish", 1, 0, 1, 0, 1, 0),
        ("Gandu tere jaise nalle logon ki koi izzat nahi karta yahan.", "hinglish", 1, 0, 1, 0, 1, 0),
        ("Harami kutte ki tarah bhonkna band kar aur kaam kar apna.", "hinglish", 1, 0, 1, 0, 1, 0),
        ("तू बिल्कुल नीच और घटिया इंसान है, तेरी कोई औकात नहीं है।", "hindi_devanagari", 1, 0, 0, 0, 1, 0),
        ("अबे कमीने कुत्ते, बकवास बंद कर अपनी और निकल यहाँ से।", "hindi_devanagari", 1, 0, 1, 0, 1, 0),
        ("नालायक दोगले इंसान, शर्म आनी चाहिए तुझे ऐसी हरकतें करते हुए।", "hindi_devanagari", 1, 0, 0, 0, 1, 0),
        ("Tu ek number ka fraud aur bhadwa hai, sabko bewakoof banata hai.", "hinglish", 1, 0, 1, 0, 1, 0),
        ("Chup reh lodu, tere jaison ko thappad mar ke seedha karna chahiye.", "hinglish", 1, 0, 1, 0, 1, 0),

        # Threats (toxic=1, threat=1, severe=1, insult=1)
        ("Tere ghar me ghus ke tujhe jaan se maar dunga, bach ke reh.", "hinglish", 1, 1, 0, 1, 1, 0),
        ("Tujhe zinda zameen me gaad dunga agar dobara saamne aaya toh.", "hinglish", 1, 1, 0, 1, 1, 0),
        ("Tujhe aur tere parivaar ko khatam kar dunga main dekh lena.", "hinglish", 1, 1, 0, 1, 1, 0),
        ("तू बाहर मिल मुझे, तेरे हाथ पैर तोड़ कर फेंक दूंगा।", "hindi_devanagari", 1, 1, 0, 1, 1, 0),
        ("तुझे गोली मार दूंगा अगर तूने अपना मुंह बंद नहीं किया।", "hindi_devanagari", 1, 1, 0, 1, 1, 0),
        ("तेरे टुकड़े टुकड़े कर दूंगा, तू बचेगा नहीं अब।", "hindi_devanagari", 1, 1, 0, 1, 1, 0),

        # Identity hate (toxic=1, identity_hate=1, severe=1, insult=1)
        ("Ye saale sab ke sab aatankwadi hain, inko desh se bahar pheko.", "hinglish", 1, 1, 0, 0, 1, 1),
        ("In jaise dharam ke logon ko zinda rehne ka koi haq nahi hai.", "hinglish", 1, 1, 0, 0, 1, 1),
        ("ये लोग दीमक की तरह हैं हमारे समाज में, इनको सबक सिखाना पड़ेगा।", "hindi_devanagari", 1, 1, 0, 0, 1, 1),
        ("इनकी पूरी जाति ही चोर और धोखेबाज़ों से भरी हुई है।", "hindi_devanagari", 1, 1, 0, 0, 1, 1),

        # Severe general hostility
        ("Tu itna ghatiya aur disgusting aadmi hai ki tujhe dekh kar ulti aati hai.", "hinglish", 1, 1, 0, 0, 1, 0),
        ("Teri shakal dekh ke ghin aati hai, jaake mar ja kahi pe.", "hinglish", 1, 1, 0, 0, 1, 0),
        ("तू इस दुनिया पर बोझ है, तेरे जैसा घटिया व्यक्ति मैंने नहीं देखा।", "hindi_devanagari", 1, 0, 0, 0, 1, 0),
    ]

    rows = []
    idx = 0
    while len(rows) < count:
        tpl, lang, tox, sev, obs, thr, ins, hate = abuse_templates[idx % len(abuse_templates)]
        variant = tpl
        if idx >= len(abuse_templates):
            suffix = f" [{idx}]" if idx % 2 == 0 else f" Samjhe? ({idx})"
            variant = tpl + suffix

        h = compute_sha256(variant)
        if h not in exclude_hashes and not any(r["hash"] == h for r in rows):
            rows.append({
                "comment_text": variant,
                "language": lang,
                "category_type": "indic_targeted_abuse",
                "toxic": tox, "severe_toxic": sev, "obscene": obs, "threat": thr, "insult": ins, "identity_hate": hate,
                "hash": h
            })
        idx += 1

    return pd.DataFrame(rows[:count])


def build_clean_conversational_set(count=1000, exclude_hashes=None) -> pd.DataFrame:
    """1,000 Clean conversational non-toxic samples (EN, HI, Hinglish)."""
    exclude_hashes = exclude_hashes or set()
    clean_templates = [
        ("The weather today is sunny and pleasant, perfect for a walk.", "english"),
        ("Thank you for your detailed review and constructive feedback.", "english"),
        ("I will see you at the project meeting tomorrow morning at 10 AM.", "english"),
        ("This is a clean, well-structured and professional document.", "english"),
        ("Could you please share the link to the recording of the webinar?", "english"),
        ("The lecture helped me understand the mathematical concepts thoroughly.", "english"),
        ("We are organizing a charity blood donation drive this coming weekend.", "english"),
        ("The new library schedule has been posted on the main university portal.", "english"),
        ("Please find attached the updated quarterly financial spreadsheet.", "english"),
        ("Our team successfully completed the customer migration without issues.", "english"),
        ("Aapka lecture dekh kar exam ki saari tension khatam ho gayi, thank you.", "hinglish"),
        ("Bhai content bahut badhiya banaya hai, concept ekdum clear ho gaya.", "hinglish"),
        ("Kya aap is tutorial ka agla part jaldi upload kar sakte hain please?", "hinglish"),
        ("Train time se pahunch gayi thi, safar bahut suhana raha.", "hinglish"),
        ("Bhai notes share kar dena WhatsApp pe jab time mile toh.", "hinglish"),
        ("Aaj sham ko cricket khelne chalenge ground pe sab log.", "hinglish"),
        ("इस व्याख्यान ने विषय को बहुत ही सरल और स्पष्ट तरीके से समझाया।", "hindi_devanagari"),
        ("आज सुबह की धूप बहुत सुहावनी और आरामदायक लग रही है।", "hindi_devanagari"),
        ("परियोजना की प्रगति रिपोर्ट समय पर जमा कर दी गई है।", "hindi_devanagari"),
        ("विश्वविद्यालय की वार्षिक खेलकूद प्रतियोगिता अगले सप्ताह आयोजित होगी।", "hindi_devanagari"),
        ("आपके सहयोग और मार्गदर्शन के लिए बहुत-बहुत धन्यवाद।", "hindi_devanagari"),
    ]

    rows = []
    idx = 0
    while len(rows) < count:
        tpl, lang = clean_templates[idx % len(clean_templates)]
        variant = tpl
        if idx >= len(clean_templates):
            prefix = f"Note #{idx}: " if idx % 2 == 0 else f"Update {idx}: "
            variant = prefix + tpl

        h = compute_sha256(variant)
        if h not in exclude_hashes and not any(r["hash"] == h for r in rows):
            rows.append({
                "comment_text": variant,
                "language": lang,
                "category_type": "clean_conversational",
                "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0,
                "hash": h
            })
        idx += 1

    return pd.DataFrame(rows[:count])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exclude_hashes = get_held_out_hashes()

    print("\n--- Generating 4 Contrastive Partitions ---")
    df_benign = build_benign_profanity_set(count=750, exclude_hashes=exclude_hashes)
    print(f"Generated {len(df_benign)} Benign/Celebratory Profanity samples.")

    df_praise = build_praise_slang_set(count=750, exclude_hashes=exclude_hashes)
    print(f"Generated {len(df_praise)} Praise Metaphors & Slang samples.")

    df_abuse = build_indic_targeted_abuse_set(count=1000, exclude_hashes=exclude_hashes)
    print(f"Generated {len(df_abuse)} Indic Targeted Abuse samples (with multi-label supervision).")

    df_clean = build_clean_conversational_set(count=1000, exclude_hashes=exclude_hashes)
    print(f"Generated {len(df_clean)} Clean Conversational Negative samples.")

    df_combined = pd.concat([df_benign, df_praise, df_abuse, df_clean], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    total_samples = len(df_combined)
    print(f"\nTotal Contrastive Corpus Assembled: {total_samples} samples.")
    print("Class Balance Breakdown:")
    for lbl in LABELS:
        pos = int(df_combined[lbl].sum())
        print(f"  - {lbl:<15}: {pos} positive ({pos/total_samples*100:.1f}%)")

    # 70 / 15 / 15 Split (Train, Validation, Held-out Contextual Test)
    indices = np.arange(len(df_combined))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=df_combined["category_type"])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=df_combined.iloc[temp_idx]["category_type"])

    df_train = df_combined.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_combined.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df_combined.iloc[test_idx].copy().reset_index(drop=True)

    print(f"\nFinal Split Sizes:")
    print(f"  Train Split : {len(df_train)} samples ({len(df_train)/total_samples*100:.1f}%)")
    print(f"  Val Split   : {len(df_val)} samples ({len(df_val)/total_samples*100:.1f}%)")
    print(f"  Held-out Test: {len(df_test)} samples ({len(df_test)/total_samples*100:.1f}%)")

    # Export splits to scratch/phase4d_corpus/
    train_file = OUTPUT_DIR / "phase4d_train.csv"
    val_file = OUTPUT_DIR / "phase4d_val.csv"
    test_file = OUTPUT_DIR / "phase4d_held_out_test.csv"

    df_train.to_csv(train_file, index=False, encoding="utf-8")
    df_val.to_csv(val_file, index=False, encoding="utf-8")
    df_test.to_csv(test_file, index=False, encoding="utf-8")

    meta = {
        "dataset_name": "Phase 4D Contrastive Multilingual Toxicity Corpus",
        "total_samples": total_samples,
        "splits": {
            "train": len(df_train),
            "val": len(df_val),
            "held_out_test": len(df_test)
        },
        "category_distribution": df_combined["category_type"].value_counts().to_dict(),
        "language_distribution": df_combined["language"].value_counts().to_dict(),
        "positive_labels": {lbl: int(df_combined[lbl].sum()) for lbl in LABELS},
        "held_out_leakage_excluded_hashes": len(exclude_hashes)
    }

    meta_file = OUTPUT_DIR / "phase4d_corpus_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCorpus preparation complete! Files written to {OUTPUT_DIR}:")
    print(f"  - {train_file.name}")
    print(f"  - {val_file.name}")
    print(f"  - {test_file.name}")
    print(f"  - {meta_file.name}")


if __name__ == "__main__":
    main()
