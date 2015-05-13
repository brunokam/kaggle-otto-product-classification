from utils import *

# Load submissions
nn = get_submission('nn.csv')
be = get_submission('be.csv')
rf = get_submission('rf.csv')
svm = get_submission('svm.csv')
xgb = get_submission('xgb.csv')

# Append scores
# predictions = nn + be + rf + svm + xgb
# predictions = nn + svm + xgb
predictions = 1.0 / 0.51 * nn + 1.0 / 0.52 * svm + 1.0 / 0.47 * xgb

# Save submission
save_submission(predictions, 'nn_svm_xgb_weighted.csv')
