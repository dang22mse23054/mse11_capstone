const BaseResponse = require('../../api/dto/BaseResponse');
const env = process.env.NEXT_PUBLIC_NODE_ENV || 'production';

class Permission {

	acceptedIpDomains(req, res, next) {
		// Aplly security middlewares
		let validIPs = process.env.ACCEPTED_IP_DOMAINS.split('|');
		if (env != 'production') {
			validIPs.push('::ffff:127.0.0.1');
		}

		let remoteIp = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
		if (validIPs.includes(remoteIp)) {
			return next();
		}

		let respObj = new BaseResponse();
		respObj.setError(403, 'Permission denied');
		return res.status(403).json(respObj);
	}

}

module.exports = Permission;