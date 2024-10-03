#!/bin/bash

cd Dounia18.github.io || exit
git remote set-url origin https://Dounia18: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@github.com/Dounia18/Dounia18.github.io
git filter-repo --path synthetic_test_webpage --invert-paths
git remote add origin https://Dounia18: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@github.com/Dounia18/Dounia18.github.io
git push --set-upstream origin main --force

cp -r ../synthetic_test_webpage ./
git add synthetic_test_webpage
git commit -m "Update PyST results"
git push origin