/**
 * Registry Resolver
 * 
 * NOTE:
 * const ResolverName = require(file_path)
 * "ResolverName" must be the same with "TypeName"
 */
const { initImport } = require('../_init');

module.exports = initImport('Category', __dirname);