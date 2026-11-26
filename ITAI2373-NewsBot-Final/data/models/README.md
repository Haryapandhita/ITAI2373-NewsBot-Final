# Trained models

| File | In git? | Description |
|---|---|---|
| `language_detector.joblib` (~1.7 MB) | ✅ | Character 1–4-gram TF-IDF + logistic regression, 6 languages, trained on 5,361 Webz sentence snippets (300 articles/language). 98.6% accuracy on 100-character inputs. |
| `newsbot2_classifier.joblib` (~36 MB) | ❌ (size, sklearn-version specific) | Soft-voting ensemble (calibrated Linear SVM + LR + Complement NB) on word+char TF-IDF; trained on the BBC train split + 600 recent English Webz articles per category. Re-created automatically by `NewsBot2.build()` in ~35 s. |
| `newsbot2_system.joblib` (~340 MB) | ❌ | Pickled `NewsBot2` object (all components + index) for fast restarts: `NewsBot2.load(path)`. Created by `bot.save()`. |

Models are loaded with `joblib`; if the installed scikit-learn version cannot unpickle a file, NewsBot retrains it
automatically. Intent classifier and topic model are re-trained on every build (< 20 s).
