const { videoService } = require('apiDir/services');

const Query = {

	getStatistic: async (obj, {options}, context, info) => {
		const { videoId, startDate, endDate } = options;
		return await videoService.getStatistic(videoId, startDate, endDate);
	},
};

module.exports = { Query };