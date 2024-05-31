class ErrorObject extends Error {
	constructor(message = 'Unknown Error', byPassSentry = false) {
		super(message);
		this.byPassSentry = byPassSentry;
		this.code = ErrorObject.DEFAULT_CODE;
		this.message = message;
		this.data = null;
	}

	setError(errorCode, message = null, data = null, byPassSentry = false) {
		this.code = errorCode;
		this.message = message;
		this.byPassSentry = byPassSentry;
		this.data = data;
		return this;
	}

	setCode(errorCode) {
		this.code = errorCode;
		return this;
	}

	setMessage(message) {
		this.message = message;
		return this;
	}

	setData(data) {
		this.data = data;
		return this;
	}

	getStack() {
		return this.stack;
	}
}

ErrorObject.DEFAULT_CODE = 500;

module.exports = ErrorObject;