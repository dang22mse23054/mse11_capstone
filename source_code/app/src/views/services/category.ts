import { ICategory, IGraphqlPagingObj } from 'interfaceDir';
import ApiRequest from './api-request';
import { Actions } from 'servDir/redux/actions';

const Service = {

	initCategoryList: (force = false) => async (dispatch, getState) => {
		let categoryList: Array<ICategory> = getState().categoryReducer.initCategoryList;

		if (force || categoryList.length == 0) {
			categoryList = await Service.getCategories();
			dispatch(Actions.CategoryAction.initCategory(categoryList));
		}
	},

	getCategories: async (keyword) => {
		const remoteUrl = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/graphql`;
		const gqlMethod = 'getCategories';
		const queryString = `
			query ${gqlMethod}  {
				${gqlMethod} {
					id
					name
				}
			}
		`;

		return await ApiRequest.sendPOST(remoteUrl, {
			operationName: gqlMethod,
			query: queryString,
			variables: {
			}

		}, (response) => {
			const obj: IGraphqlPagingObj<ICategory> = response.data.data[gqlMethod];
			return obj;

		}, (error) => {
			console.error(error);
			return null;
		});
	}
};

export default Service;