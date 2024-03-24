import { VideoInput, VideoStatusInput } from 'inputModelDir';
import { /* ICsvReport,  */ IGraphqlPagingObj, IVideo } from 'interfaceDir';
import ApiRequest from './api-request';

const getQuery = (type) => `{
	id
	title
	path

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

	updateStatus: async (taskId, status, updatedAt, searchOptions) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'changeVideoStatus';
		const queryString = `
			mutation ${gqlMethod} ($task: VideoStatusInput!, $searchOptions: SearchVideoInput) {
				${gqlMethod}(task: $task, searchOptions: $searchOptions) {
					id
					hasErr
					message
				}
			}
		`;
		return await ApiRequest.sendPOST(remoteUrl, {
			operationName: gqlMethod,
			query: queryString,
			variables: {
				task: new VideoStatusInput(taskId, status, updatedAt),
				searchOptions
			}

		}, (response) => {
			const obj = response.data.data[gqlMethod];
			return obj;

		}, (error) => {
			console.error(`Error when calling ${gqlMethod}: `, error);
			return null;
		});
	},

	updateSetting: async (task) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'updateVideo';
		const queryString = `
			mutation ${gqlMethod} ($task: VideoInput!) {
				${gqlMethod}(task: $task) {
					id
					hasErr
					message
				}
			}
		`;
		return await ApiRequest.sendPOST(remoteUrl, {
			operationName: gqlMethod,
			query: queryString,
			variables: {
				task: new VideoInput(task),
			}

		}, (response) => {
			const obj = response.data.data[gqlMethod];
			return obj;

		}, (error) => {
			console.error(error);
			return null;
		});
	},
};

export default Service;