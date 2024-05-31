const User = require('modelDir/User');
const UserBO = require('../business/UserBO');
const BaseLoader = require('./_base');
const DataLoader = require('dataloader');

module.exports = class UserLoader extends BaseLoader {

	constructor() {
		super(User);
		this.userBO = new UserBO();

		this._loadByUids = new DataLoader(this.userBO._loadByUids);
	}

	loadByUids = async (uids, isDeleted = null) => {
		const users = await this._loadByUids.loadMany(uids.map(uid => uid.toString().toLowerCase()));
		//TODO isDeleted ? user.deletedAt : !user.deletedAt)
		return isDeleted != null ? users.filter(user => isDeleted ? user.deletedAt : !user.deletedAt) : users;
	}
	loadByUid = async (uid) => this._loadByUids.load(uid.toString().toLowerCase())

	// custom `clearAll`
	clearAll = () => {
		this.defaultClearAll();
		this._loadByUids.clearAll();
	}

};