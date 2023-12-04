from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# HTML form to choose a video and an int variable
@app.route('/')
def index():
    return render_template('index.html')

# Process video endpoint
@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the int variable from the form
    int_variable = int(request.form['int_variable'])

    # Process the video (you can add your video processing logic here)

    # For demonstration purposes, just return a simple response
    return f"Video processing completed. Int Variable: {int_variable}"

if __name__ == '__main__':
    app.run(debug=True)
