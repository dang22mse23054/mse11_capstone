const BaseResponse = require('./BaseResponse');
const { ActionStatus } = require('commonDir/constants/status');

module.exports = class CAUserListResp extends BaseResponse {
	constructor() {
		super();
		this.totalPages = 0;
		this.totalRecords = 0;
		this.recordsPerPage = process.env.DEFAULT_RECORDS_PER_PAGE;
		this.data = [];
	}

	addToList(user) {
		let _list = this.data;

		let key = `${user.uid}${user.normalize}${user.email}`;
		_list.push({
			key: key,
			value: {
				id: user.id,
				uid: user.uid,
				fullname: user.fullname,
				kana: user.kana,
				email: user.email,
				division2: user.division2,
				division3: user.division3,
				division4: user.division4,
				division5: user.division5,
				division6: user.division6,
				status: ActionStatus.SKIP
			}
		});
	}

	addData(userList) {
		if (userList) {
			if (Array.isArray(userList)) {
				for (let user of userList) {
					this.addToList(user);
				}
			} else if (typeof userList === 'object') {
				this.addToList(userList);
			}
		}
	}
};