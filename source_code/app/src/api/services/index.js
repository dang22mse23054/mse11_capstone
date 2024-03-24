const UserService = require('./UserService');
const VideoService = require('./VideoService');
const CategoryService = require('./CategoryService');

exports.userService = new UserService();
exports.videoService = new VideoService();
exports.categoryService = new CategoryService();
