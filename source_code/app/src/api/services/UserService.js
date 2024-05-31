// const { ErrorCodes } = require('commonDir/constants');
const UserBO = require('../db/business/UserBO');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const { SystemCache } = require('apiDir/db/graphql');

module.exports = class UserService {

	// using loader to get user info
	getUserByUid = async (uid) => {
		try {
			if (Array.isArray(uid)) {
				return SystemCache.user.loader.loadByUids(uid); 

			} else {
				return SystemCache.user.loader.loadByUid(uid);
			}
		} catch (error) {
			log.error(error);
		}
		return null;
	}

	// using loader to get user(s) info
	getUsersByUids = async (uids) => this.getUserByUid(uids)

	searchCAUsers = async (keyword, withDeleted, nextCursor, limit = 10) => {
		let userBO = new UserBO();
		try {
			if (keyword && keyword != '') {
				return userBO.search(keyword, withDeleted, nextCursor, limit);
			}
		} catch (err) {
			log.error(err);
		}
		return null;
	}

	updateCAUser = async (user) => {
		let userBO = new UserBO();
		try {
			await userBO.update(user);
			return true;
		} catch (err) {
			log.error(err);
		}
		return false;
	}

	updateLastAccess = (uid) => {
		let userBO = new UserBO();
		userBO.setLastAccess(uid);
	}
};