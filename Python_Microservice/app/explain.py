import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from .schema import FactorContribution

def explain_prediction(model, instance: np.ndarray, feature_names):
    """Compute SHAP + LIME for transparency demo."""
    explainer = shap.KernelExplainer(model.predict_proba, np.random.rand(10, len(instance)))
    shap_values = explainer.shap_values(np.array([instance]))[1]

    lime_explainer = LimeTabularExplainer(
        training_data=np.random.rand(100, len(instance)),
        mode="classification",
        feature_names=feature_names
    )
    lime_exp = lime_explainer.explain_instance(instance, model.predict_proba, num_features=3)

    shap_top = [FactorContribution(feature=f"f{i}", shap_value=float(v)) for i, v in enumerate(shap_values[:3])]
    lime_top = [FactorContribution(feature=str(name), shap_value=float(weight)) for name, weight in lime_exp.as_list()]
    return {"shap": shap_top, "lime": lime_top}
