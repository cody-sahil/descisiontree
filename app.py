import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Decision Tree model
with open('decision.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with a modern, attractive Tailwind CSS layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Placement Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen py-10 px-4 sm:px-6 lg:px-8">
    <div class="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100">
        <!-- Header -->
        <div class="bg-gradient-to-r from-blue-600 to-indigo-700 py-6 px-8 text-white">
            <h1 class="text-2xl sm:text-3xl font-bold tracking-tight">Campus Placement Predictor</h1>
            <p class="text-blue-100 text-sm mt-1">Enter student academic and professional details to predict placement status.</p>
        </div>

        <!-- Form Body -->
        <form method="POST" class="p-8 space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- Gender -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Gender</label>
                    <select name="gender" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <!-- Work Experience -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Work Experience</label>
                    <select name="workex" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <!-- SSC Percentage -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">SSC (10th) Percentage (%)</label>
                    <input type="number" step="0.01" name="ssc_p" placeholder="e.g. 67.0" required class="w-full rounded-lg border border-slate-300 py-2 px-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- SSC Board -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">SSC Board</label>
                    <select name="ssc_b" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="1">Central</option>
                        <option value="0">Others</option>
                    </select>
                </div>

                <!-- HSC Percentage -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">HSC (12th) Percentage (%)</label>
                    <input type="number" step="0.01" name="hsc_p" placeholder="e.g. 75.5" required class="w-full rounded-lg border border-slate-300 py-2 px-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- HSC Board -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">HSC Board</label>
                    <select name="hsc_b" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="1">Central</option>
                        <option value="0">Others</option>
                    </select>
                </div>

                <!-- HSC Stream -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">HSC Stream</label>
                    <select name="hsc_s" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="0">Science</option>
                        <option value="1">Commerce</option>
                        <option value="2">Arts</option>
                    </select>
                </div>

                <!-- Degree Percentage -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Degree Percentage (%)</label>
                    <input type="number" step="0.01" name="degree_p" placeholder="e.g. 65.0" required class="w-full rounded-lg border border-slate-300 py-2 px-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Degree Type -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Degree Type (Under Graduate)</label>
                    <select name="degree_t" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="0">Sci&Tech</option>
                        <option value="1">Comm&Mgmt</option>
                        <option value="2">Others</option>
                    </select>
                </div>

                <!-- Etest Percentage -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Employability Test Score (etest_p)</label>
                    <input type="number" step="0.01" name="etest_p" placeholder="e.g. 70.0" required class="w-full rounded-lg border border-slate-300 py-2 px-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

                <!-- Specialisation -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">MBA Specialisation</label>
                    <select name="specialisation" required class="w-full rounded-lg border border-slate-300 py-2.5 px-3 bg-white text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                        <option value="0">Mkt&Fin</option>
                        <option value="1">Mkt&HR</option>
                    </select>
                </div>

                <!-- MBA Percentage -->
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">MBA Percentage (%)</label>
                    <input type="number" step="0.01" name="mba_p" placeholder="e.g. 58.8" required class="w-full rounded-lg border border-slate-300 py-2 px-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                </div>

            </div>

            <!-- Submit Button -->
            <div class="pt-4">
                <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 shadow-md">
                    Predict Placement Status
                </button>
            </div>
        </form>

        <!-- Results Section -->
        {% if prediction is not none %}
        <div class="mx-8 mb-8 p-6 rounded-xl border text-center 
            {% if prediction == 1 or prediction == 'Placed' %} bg-emerald-50 border-emerald-200 text-emerald-800 
            {% else %} bg-rose-50 border-rose-200 text-rose-800 {% endif %}">
            <h2 class="text-lg font-semibold">Prediction Result</h2>
            <p class="text-2xl font-bold mt-1">
                {% if prediction == 1 or prediction == 'Placed' %}
                    🎉 Placed
                {% else %}
                    ⚠️ Not Placed
                {% endif %}
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the exact order expected by the model:
            # ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']
            features = [
                float(request.form['gender']),
                float(request.form['ssc_p']),
                float(request.form['ssc_b']),
                float(request.form['hsc_p']),
                float(request.form['hsc_b']),
                float(request.form['hsc_s']),
                float(request.form['degree_p']),
                float(request.form['degree_t']),
                float(request.form['workex']),
                float(request.form['etest_p']),
                float(request.form['specialisation']),
                float(request.form['mba_p'])
            ]
            
            final_features = np.array([features])
            pred = model.predict(final_features)[0]
            prediction = pred
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = None

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)