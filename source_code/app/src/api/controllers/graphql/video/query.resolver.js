const { videoService } = require('apiDir/services');

const Query = {
	searchVideos: async (obj, {options, cursor, limit}, context, info) => {
		const data = await videoService.searchVideos(options, cursor, limit);

		return {
			edges: data.nodes.map(node => ({
				...node,
				node: {
					__typename: 'Video',
					...node.data
				},
			})),
			pageInfo: {
				...data.pageInfo,
				limit,
				lastCursor: cursor ? cursor.nextCursor || cursor.prevCursor : null
			}
		};
	},

	getVideo: async (obj, {id}, context, info) => {
		return await videoService.getVideo({id});
	},
};

module.exports = { Query };