rm -f hw5_code.zip
zip -r hw5_code.zip dynamic_programming/*.ipynb dynamic_programming/*.py dynamic_programming/*.pdf q_learning/*.pdf q_learning/*.py q_learning/*/*.py q_learning/*.ipynb -x ".*" "__MACOSX"
echo "Submission file hw5_code.zip created."
