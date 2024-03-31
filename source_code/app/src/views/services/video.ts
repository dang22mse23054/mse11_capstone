import { VideoInput } from 'inputModelDir';
import { /* ICsvReport,  */ IGraphqlPagingObj, IVideo } from 'interfaceDir';
import ApiRequest from './api-request';

const getQuery = (type) => `{
	id
	title
	refFileName
	refFilePath

	status
	isEnabled
	deletedAt
	
	categoryIds
	categories {
		id
		name
	}

	createdAt
}`;

const Service = {

	searchVideos: async (options, cursor, limit) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'searchVideos';
		const queryString = `
			query ${gqlMethod} ($options: SearchVideoInput, $cursor: CursorInput, $limit: Int) {
				${gqlMethod}(options: $options, cursor: $cursor, limit: $limit) {
					edges {
						node {
						  __typename
						  id
						  ... on Video ${getQuery()}
						}
					}
					pageInfo {
						total
						limit
						next
						previous
						lastCursor
					}
				}
			}
		`;

		return await ApiRequest.sendPOST(remoteUrl, {
			operationName: gqlMethod,
			query: queryString,
			variables: {
				options,
				cursor,
				limit
			}
		}, (response) => {
			if (response.errors) {
				return null;
			}

			const obj: IGraphqlPagingObj<IVideo> = response.data.data[gqlMethod];
			return {
				list: obj.edges.map(edge => {
					const video = edge.node;
					delete video['__typename'];

					return video;
				}),
				pageInfo: obj.pageInfo
			};

		}, (error) => {
			console.error(error);
			return null;
		});
	},

	getVideo: async (id: number) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'getVideo';
		const queryString = `
			query ${gqlMethod} ($id: Int!) {
				${gqlMethod}(id:$id) ${getQuery('detail')}
			}
		`;

		return await ApiRequest.sendPOST(remoteUrl, {
			operationName: gqlMethod,
			query: queryString,
			variables: {
				id
			}
		}, (response) => {
			const obj: IVideo = response.data.data[gqlMethod];
			return obj;

		}, (error) => {
			console.error(error);
			return null;
		});
	},

	/**
	 * 
	 * @param {VideoInput} video 
	 * @returns 
	 */
	insertOrUpdateVideo: async (video) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'insertOrUpdateVideo';
		const queryString = `
			mutation ${gqlMethod} ($video: VideoInput!) {
				${gqlMethod}(video: $video) {
					id
					hasErr
					message
				}
			}
		`;

		return await ApiRequest.sendPOST(remoteUrl, {
			//add operationName and banned API cause stress server
			operationName: gqlMethod,
			query: queryString,
			variables: { video }

		}, (response) => {
			const obj = response.data.data[gqlMethod];
			return obj;

		}, (error) => {
			console.error(error);
			throw error.response?.data?.errors[0];
		});
	},

	updateVideoStatus: async (video, isDel = false) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'updateVideoStatus';
		const queryString = `
			mutation ${gqlMethod} ($video: VideoEnabledInput!) {
				${gqlMethod}(video: $video ${isDel ? ', isDel: true' : ''}) 
			}
		`;

		return await ApiRequest.sendPOST(remoteUrl, {
			operationName: gqlMethod,
			query: queryString,
			variables: { video }
		}, (response) => {
			const isSuccess: boolean = response.data.data[gqlMethod];
			return isSuccess;

		}, ((error) => {
			console.error(error);
			return null;
		}));
	},

	convertVideo: async (video = {} as IVideo) => {

		// parse category
		video.categoryIds = [];
		video.categories = video.categories?.map((item) => {
			video.categoryIds.push(Number(item.id));
			return {
				key: `${item.id}${item.name}`,
				value: { ...item }
			};
		});

		return video;
	}
};

export default Service;