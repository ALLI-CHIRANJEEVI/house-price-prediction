
from flask import Flask, request, send_file, jsonify
import pandas as pd
import joblib
import datetime
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_csv():
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    import traceback

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    model_path = os.path.join(PROCESSED_FOLDER, 'model.joblib')
    pipeline_path = os.path.join(PROCESSED_FOLDER, 'pipeline.joblib')

    try:
        if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
            housing = pd.read_csv(filepath)
            if "median_house_value" not in housing.columns:
                return jsonify({"error": "Training requires 'median_house_value' column"}), 400

            housing['income_cat'] = pd.cut(
                housing["median_income"],
                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1, 2, 3, 4, 5]
            )
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(housing, housing['income_cat']):
                strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)

            housing = strat_train_set.copy()
            housing_labels = housing["median_house_value"].copy()
            housing = housing.drop("median_house_value", axis=1)

            num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
            cat_attribs = ["ocean_proximity"]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", cat_pipeline, cat_attribs)
            ])
            housing_prepared = full_pipeline.fit_transform(housing)
            model = RandomForestRegressor()
            model.fit(housing_prepared, housing_labels)

            joblib.dump(model, model_path)
            joblib.dump(full_pipeline, pipeline_path)
        else:
            model = joblib.load(model_path)
            full_pipeline = joblib.load(pipeline_path)

        input_df = pd.read_csv(filepath)
        if "median_house_value" in input_df.columns:
            input_features = input_df.drop("median_house_value", axis=1)
        else:
            input_features = input_df.copy()

        input_prepared = full_pipeline.transform(input_features)
        predictions = model.predict(input_prepared)
        output_df = input_df.copy()
        output_df["predicted_house_value"] = predictions

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'output_{timestamp}_{filename}'
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        output_df.to_csv(output_path, index=False)

        return jsonify({"filename": output_filename})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    output_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(output_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(output_path, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
