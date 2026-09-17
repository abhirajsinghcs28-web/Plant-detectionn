const express = require('express');
const router = express.Router();
const upload = require('../middleware/uploadMiddleware');
const {
  predictCropLeaf,
  getPredictions,
  deletePrediction,
} = require('../controllers/predictController');

// Route for leaf disease & severity prediction
router.post('/predict', upload.single('image'), predictCropLeaf);

// Route for retrieving past predictions
router.get('/predictions', getPredictions);

// Route for deleting a prediction record
router.delete('/predictions/:id', deletePrediction);

module.exports = router;
