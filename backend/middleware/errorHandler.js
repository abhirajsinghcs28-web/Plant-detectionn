const multer = require('multer');

/**
 * Centralized error handler middleware
 */
function errorHandler(err, req, res, next) {
  console.error('Error encountered:', err.message || err);

  // Multer specific errors
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: 'Uploaded image is too large. Maximum file size allowed is 10MB.',
      });
    }
    return res.status(400).json({
      success: false,
      error: `File upload error: ${err.message}`,
    });
  }

  // Known validation errors
  if (err.statusCode) {
    return res.status(err.statusCode).json({
      success: false,
      error: err.message,
    });
  }

  // General server error
  return res.status(500).json({
    success: false,
    error: err.message || 'Prediction failed. Please try again.',
  });
}

module.exports = errorHandler;
