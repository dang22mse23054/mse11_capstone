const UserService = require('./UserService');
const VideoService = require('./VideoService');
const CategoryService = require('./CategoryService');
const S3Service = require('./S3Service');

exports.userService = new UserService();
exports.videoService = new VideoService();
exports.categoryService = new CategoryService();
exports.s3Service = new S3Service();
