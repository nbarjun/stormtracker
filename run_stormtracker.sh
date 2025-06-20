# The code that runs the StormTracker and updates
# the latest map

pip install -r requirements.txt
python detect_latest_storms.py
git config user.name "github-actions"
git config user.email "github-actions@github.com"
git add latest_storms.html
git commit -m "Updated latest storms"
git push