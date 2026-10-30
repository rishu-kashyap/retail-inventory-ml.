"""
Optimized Retail Inventory Management System (ML & Decision Support)
Flask Web Application & REST API
Student: Rishu Kumar (Roll: 2400320230102)
"""
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from src.model_service import InventoryModelService

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "retail-inventory-secret-key-2026")

# Initialize model service
service = InventoryModelService.get_instance()


@app.route("/")
def index():
    """Executive Dashboard with Live Analytics, KPIs, and Charts."""
    data = service.get_dashboard_data()
    return render_template(
        "index.html",
        summary=data["summary"],
        dept_summary=data["dept_summary"],
        comparison=data["comparison"],
        leaderboard=data["leaderboard"][:5],
        top_risk_items=data["top_risk_items"],
        overstock_items=data["overstock_items"],
        student_name="Rishu Kumar",
        student_roll="2400320230102"
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Interactive Demand Forecast and Reorder Recommendation Engine."""
    stores = service.get_stores()
    items = service.get_items()[:60]

    result = None
    selected_store = stores[0] if stores else "CA_1"
    selected_item = items[0] if items else "FOODS_3_120"
    current_inventory = 70
    lead_time = 5
    unit_cost = 2.50
    unit_price = 4.50
    service_level = 0.95
    in_promo = 0
    is_weekend = 0

    if request.method == "POST":
        selected_store = request.form.get("store_id", selected_store)
        selected_item = request.form.get("item_id", selected_item)
        current_inventory = float(request.form.get("current_inventory", 70))
        lead_time = int(request.form.get("lead_time_days", 5))
        unit_cost = float(request.form.get("unit_cost", 2.50))
        unit_price = float(request.form.get("unit_price", 4.50))
        service_level = float(request.form.get("service_level", 0.95))
        in_promo = 1 if request.form.get("in_promo") == "1" else 0
        is_weekend = 1 if request.form.get("is_weekend") == "1" else 0

        result = service.predict_and_optimize(
            store_id=selected_store,
            product_id=selected_item,
            current_inventory=current_inventory,
            lead_time_days=lead_time,
            unit_cost=unit_cost,
            unit_price=unit_price,
            target_service_level=service_level,
            in_promo=in_promo,
            is_weekend=is_weekend
        )
    else:
        # Default run matching Example on Page 3 of Project Validation Document
        result = service.predict_and_optimize(
            store_id=selected_store,
            product_id=selected_item,
            current_inventory=current_inventory,
            lead_time_days=lead_time,
            unit_cost=unit_cost,
            unit_price=unit_price,
            target_service_level=service_level,
            in_promo=0,
            is_weekend=0
        )

    return render_template(
        "predict.html",
        stores=stores,
        items=items,
        selected_store=selected_store,
        selected_item=selected_item,
        current_inventory=current_inventory,
        lead_time=lead_time,
        unit_cost=unit_cost,
        unit_price=unit_price,
        service_level=service_level,
        in_promo=in_promo,
        is_weekend=is_weekend,
        result=result
    )


@app.route("/inventory")
def inventory():
    """Live Inventory Status, Risk Detection, and Reorder Ledger."""
    store_filter = request.args.get("store", "ALL")
    risk_filter = request.args.get("risk", "ALL")
    search_query = request.args.get("search", "").strip().lower()

    ledger = service.inventory_ledger

    filtered = []
    for item in ledger:
        if store_filter != "ALL" and item["store_id"] != store_filter:
            continue
        if risk_filter != "ALL" and item["risk_level"] != risk_filter:
            continue
        if search_query:
            if (search_query not in item["product_id"].lower() and
                search_query not in item["store_id"].lower() and
                search_query not in item["dept"].lower()):
                continue
        filtered.append(item)

    stores = service.get_stores()

    # Counts
    counts = {
        "ALL": len(ledger),
        "STOCKOUT_RISK": sum(1 for i in ledger if i["risk_level"] == "STOCKOUT_RISK"),
        "LOW_STOCK": sum(1 for i in ledger if i["risk_level"] == "LOW_STOCK"),
        "OVERSTOCK_RISK": sum(1 for i in ledger if i["risk_level"] == "OVERSTOCK_RISK"),
        "NORMAL": sum(1 for i in ledger if i["risk_level"] == "NORMAL"),
    }

    return render_template(
        "inventory.html",
        items=filtered,
        stores=stores,
        store_filter=store_filter,
        risk_filter=risk_filter,
        search_query=search_query,
        counts=counts,
        total_filtered=len(filtered)
    )


@app.route("/models")
def models_evaluation():
    """Model Comparison, Evaluation Metrics & Leaderboard."""
    data = service.get_dashboard_data()
    feature_importance = []
    feat_csv = service.REPORTS_DIR / "feature_importance.csv" if hasattr(service, "REPORTS_DIR") else Path("reports/feature_importance.csv")
    if feat_csv.exists():
        import pandas as pd
        df_feat = pd.read_csv(feat_csv).head(15)
        feature_importance = df_feat.to_dict(orient="records")

    return render_template(
        "models.html",
        leaderboard=data["leaderboard"],
        comparison=data["comparison"],
        feature_importance=feature_importance
    )


@app.route("/assistant")
def assistant():
    """Natural Language AI Inventory Decision Support Layer."""
    return render_template("assistant.html")


# =========================================================================
# REST API ENDPOINTS
# =========================================================================

@app.route("/api/predict", methods=["POST", "GET"])
def api_predict():
    """
    REST API for demand forecasting and reorder recommendation.
    Accepts JSON body or query parameters.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form.to_dict()
    else:
        data = request.args.to_dict()

    store_id = data.get("store_id", "CA_1")
    product_id = data.get("product_id", data.get("item_id", "FOODS_3_120"))
    current_inventory = float(data.get("current_inventory", 70))
    lead_time = int(data.get("lead_time_days", 5))
    unit_cost = float(data.get("unit_cost", 2.50))
    unit_price = float(data.get("unit_price", 4.50))
    service_level = float(data.get("service_level", 0.95))
    in_promo = 1 if str(data.get("in_promo", "0")) in ["1", "true", "True"] else 0
    is_weekend = 1 if str(data.get("is_weekend", "0")) in ["1", "true", "True"] else 0

    res = service.predict_and_optimize(
        store_id=store_id,
        product_id=product_id,
        current_inventory=current_inventory,
        lead_time_days=lead_time,
        unit_cost=unit_cost,
        unit_price=unit_price,
        target_service_level=service_level,
        in_promo=in_promo,
        is_weekend=is_weekend
    )

    return jsonify({"success": True, "data": res})


@app.route("/api/inventory", methods=["GET"])
def api_inventory():
    """Returns live inventory ledger filtered by risk or store."""
    risk = request.args.get("risk")
    store = request.args.get("store")
    limit = int(request.args.get("limit", 100))

    items = service.inventory_ledger
    if risk:
        items = [i for i in items if i["risk_level"] == risk]
    if store:
        items = [i for i in items if i["store_id"] == store]

    return jsonify({"success": True, "count": len(items), "items": items[:limit]})


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Returns aggregated executive KPIs."""
    dash = service.get_dashboard_data()
    return jsonify({"success": True, "summary": dash["summary"]})


@app.route("/api/assistant", methods=["POST"])
def api_assistant():
    """Handles natural language questions regarding inventory risk and decisions."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"success": False, "answer": "Please provide a question."}), 400

    reply = service.query_assistant(query)
    return jsonify({"success": True, "query": query, "answer": reply})


@app.route("/api/sku-profile/<store_id>/<item_id>")
def api_sku_profile(store_id, item_id):
    """Auto-fill helper for SKU dropdown change in UI."""
    prof = service.get_sku_profile(store_id, item_id)
    return jsonify(prof)


@app.route("/health")
def health():
    """Health check endpoint for Render / Docker / Railway."""
    return jsonify({"status": "healthy", "service": "retail-inventory-ml", "version": "1.0.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
