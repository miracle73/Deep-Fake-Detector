export function isValidFile(file: Express.Multer.File): boolean {
  const allowedTypes = ['video/mp4', 'image/jpeg'];
  const maxSizeMB = 50;
  return (
    allowedTypes.includes(file.mimetype) && file.size < maxSizeMB * 1024 * 1024
  );
}
