import multer from 'multer';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 20 * 1024 * 1024, // 20MB max
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['video/mp4', 'video/mpeg', 'video/quicktime'];
    if (!allowedTypes.includes(file.mimetype)) {
      return cb(new Error('Only video files are allowed'));
    }
    cb(null, true);
  },
});

export default upload;
