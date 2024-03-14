import Encoding from 'encoding-japanese';
import EmojisConverter from 'neato-emoji-converter';

const emojisConverter = new EmojisConverter();
const units = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

const Utils = {
	unicodeToEmoji: (value) => emojisConverter.replaceShortcodes(value),
	emojiToUnicode: (value) => emojisConverter.replaceUnicode(value),
	removeNullData: (obj) => {
		const newObj = { ...obj };
		Object.keys(newObj).forEach((key) => (newObj[key] == null) && delete newObj[key]);
		return newObj;
	},

	isNotEmpty: (value) => {
		if (typeof value == typeof true) {
			return true;
		}

		return value ? true : false;
	},

	parseUint8Array: (content, charset: 'SJIS' | 'UTF8') => {
		const uniArray = Encoding.stringToCode(charset == 'SJIS' ? Utils.emojiToUnicode(content) : content);
		const sjisArray = Encoding.convert(uniArray, {
			from: 'UNICODE',
			to: charset,
			bom: true // With BOM
		});
		// const sjisString = Encoding.codeToString(sjisArray);
		const unit8Array = new Uint8Array(sjisArray);
		return unit8Array;
	},

	outputCsv: (filename = '', charset: 'SJIS' | 'UTF8', obj: Array<any> | string) => {
		let content = '';
		if (typeof (obj) === 'string') {
			content = obj;
		} else if (obj) {
			obj.forEach(function (infoArray, index) {
				const dataString = infoArray.join(',');
				content += index < obj.length ? dataString + '\n' : dataString;
			});
		}

		const a = document.createElement('a');
		const mimeType = charset ? ('text/csv,charset:' + charset) : 'application/octet-stream';

		if (window.navigator && window.navigator.msSaveBlob) { // IE10
			navigator.msSaveBlob(new Blob([Utils.parseUint8Array(content, charset)], {
				type: mimeType
			}), filename);
		} else if (URL && 'download' in a) { //html5 A[download]
			a.href = URL.createObjectURL(new Blob([Utils.parseUint8Array(content, charset)], {
				type: mimeType
			}));
			a.setAttribute('download', filename);
			a.setAttribute('charset', charset);
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);

		} else {
			location.href = 'data:application/octet-stream,' + encodeURIComponent(content); // only this mime type is supported
		}

	},

	isTheSameTime: (t1, t2) => new Date(t1).getTime() !== new Date(t2).getTime(),

	toUrlParams: (data) => {
		const params = new URLSearchParams();

		Object.entries(data).forEach(([key, value]) => {
			if (Array.isArray(value)) {
				value.forEach(value => params.append(key, value));
			} else {
				params.append(key, value);
			}
		});

		return params;
	},

	openNewTab: (url, title = '') => {
		const tab = window.open(url, '_blank');
		// tab.document.body.innerHTML = "Please wait...";
		tab.opener = null;
		setTimeout(() => {
			tab.document.title = title;
		}, 100);
	},

	isEqualArray: (oldArray, newArray) => {
		const isOldArray = Array.isArray(oldArray);
		const isNewArray = Array.isArray(newArray);
		// if all old and new is Array type
		if (isOldArray && isNewArray) {
			const mergedSet = new Set([...oldArray, ...newArray]);

			if (mergedSet.size > 0) {
				if (mergedSet.size == oldArray.length && mergedSet.size == newArray.length) {
					return true;
				}
				return false;
			}
		} else if (isOldArray || isNewArray) {
			// if old OR new is Array type
			return false;
		}

		// eg: undefined or old=new=[]
		return true;
	},

	niceBytes: (x) => {
		let l = 0;
		let n = parseInt(x, 10) || 0;
		while (n >= 1024 && ++l) {
			n = n / 1024;
		}
		return (n.toFixed(n < 10 && l > 0 ? 1 : 0) + ' ' + units[l]);
	},

	strNormalize: (str, options = { removeSpace: false, toLowerCase: false, trim: true, toFullWidth: true, replaceSpaceWith: ' ', removeSpecialChar: false }) => {
		if (str) {
			str = str.replace('\u3000', ' ');
			if (options.toFullWidth != false) { str = str.normalize('NFKC'); }
			if (options.removeSpace == true) { str = str.replace(/\s{1,}/g, ''); }
			if (options.replaceSpaceWith) { str = str.replace(/\s{1,}/g, options.replaceSpaceWith); }
			if (options.removeSpecialChar != false) { str = str.replace(/[&/\\#,+()$~%.。・〜'":*?<>{}]/g, ''); }
			if (options.toLowerCase != false) { str = str.toLowerCase(); }
			if (options.trim != false) { str = str.trim(); }
		}
		return str;
	},

	/**
	 * 
	 * @param fileName 
	 * @param clientNames 
	 * @param minClient : minimum of client. 0 is uncheck minimum
	 * @param maxClient : maximum of client. 0 is uncheck maximum
	 * @returns boolean
	 */
	hasClentInName: (fileName: string, clientNames: Array<string> = [], { minClient = 0, maxClient = 0 } = {}) => {
		let result = true;
		// check has file name
		if (!fileName) {
			return false;
		}
		fileName = Utils.strNormalize(fileName);

		let countClient = 0;

		for (let index = 0; index < clientNames.length; index++) {
			const clientName = Utils.strNormalize(clientNames[index]);
			// check fileName has clientName
			if (fileName.includes(clientName)) {
				countClient++;
			}
		}

		// check minimum
		if (minClient == 0) {
			result = true;
		} else {
			result = minClient <= countClient;
		}

		// check maximum
		if (result) {
			if (maxClient == 0) {
				result = true;
			} else {
				result = maxClient >= countClient;
			}


		}

		return result;
	},

};

export default Utils;