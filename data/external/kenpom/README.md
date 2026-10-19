# KenPom daily paste

1. On kenpom.com (signed in), select the whole ratings table and copy it.
2. On GitHub, open `data/external/kenpom/paste.txt`, click the pencil (edit),
   select all, paste, and click **Commit changes**. (First time: use
   **Add file → Create new file** and name it `data/external/kenpom/paste.txt`.)

The commit starts a refresh. The model converts the paste to `teams.csv`,
stamps the date, and uses it for Cal's Model 2 (and blends it into the Edge
Board ratings). A paste older than 3 days is ignored, and the models fall back
to the Edge Board's own ratings. Teams it can't match also fall back.
