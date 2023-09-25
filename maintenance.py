from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def maintenance_page():
    return render_template('maintenance.html')

# Route to redirect to the maintenance page for /docs, /api/books, and /api/recommendations
@app.route('/docs')
@app.route('/api/books')
@app.route('/api/recommendations')
def redirect_to_maintenance():
    return redirect(url_for('maintenance_page'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
