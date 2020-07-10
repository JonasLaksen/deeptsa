FILE=$1
cp $FILE "server_${FILE}" -R
git add server_results/
git commit -m "Automatic commit, adding server result ${FILE}"
git push origin HEAD
