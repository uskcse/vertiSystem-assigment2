from flask import Flask, request, render_template, jsonify, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
from app.model import (
    preprocess_data,
    train_model,
    predict_model,
)  # Assuming this file contains the code you provided

app = Flask(__name__)

# Configuration for file upload
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Check if the file is a CSV
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Read the uploaded CSV file into a DataFrame
        try:
            data = pd.read_csv(file_path)

            # Pass the DataFrame to preprocess_data function
            processed_data = preprocess_data(data)

            # Optionally, you could also train the model on the uploaded data
            # train_model(processed_data)  # Uncomment if you want to retrain on each upload

            # Use the trained model to make predictions
            
            predicted_data = predict_model(processed_data)

            # Save the processed and predicted data to a CSV file
            output_file = "predicted_data.csv"
            predicted_data.to_csv(output_file, index=False)

            # Return the predicted data to be displayed
            return render_template(
                "index.html",
                tables=[predicted_data.to_html(classes="data")],
                titles=predicted_data.columns.values,
                download_link=output_file,
            )

        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400


@app.route("/download/<filename>")
def download_file(filename):
    """
    Provides a downloadable link for the processed CSV file.
    """
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)