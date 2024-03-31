const { videoService } = require('apiDir/services');
// const { Common, VideoStatus, VideoActionType } = require('commonDir/constants');

const Mutation = {

	insertOrUpdateVideo: async (obj, { video }, { userInfo }, info) => {
		// Graphql type will check whether required fields are missing or not.
		const response = await videoService.insertOrUpdate(video, userInfo);
		if (response?.id) {
			return response;
		}
		// UI service will take null as signal for displaying error toast
		return null;
	},

	updateVideoStatus: async (obj, { video, isDel }, { userInfo }, info) => {
		return await videoService.updateVideoStatus(video, isDel) != null;
	},
};

module.exports = { Mutation };