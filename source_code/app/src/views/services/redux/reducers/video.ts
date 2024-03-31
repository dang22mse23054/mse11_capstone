import { ActionTypes } from '../actions';
import { IVideoActionObj, IVideoSearchAO, IVideoSettingAO, IActionObj } from '../actions/action-object';

export interface IVideoReducer {
	// using for update Video task
	setting: IVideoActionObj
	list: IVideoSearchAO
}

const initialState: IVideoReducer = {
	setting: {
		id: 0,
		isLoading: true,
	},
	list: {
		searchOpts: {},
		videoList: [],
		pageInfo: {
			currentPage: 0,
			limit: 1,
			total: 0
		},
		isLoading: true,
	},
};

const handler = (state = initialState, action: IActionObj) => {
	switch (action.type) {
		//-------- Video list action --------//
		case ActionTypes.Video.listPage.search.show:
			const { searchInput, videoList, pageInfo } = (action as IVideoSearchAO);
			return showList(state, searchInput, videoList, pageInfo);
			break;

		case ActionTypes.Video.listPage.setIsLoading:
			return setListLoading(state, (action as IVideoSearchAO).isLoading);
			break;
		
		//-------- Video setting action --------//
		case ActionTypes.Video.settingPage.setInfo:
			return setVideo(state, action);

		case ActionTypes.Video.settingPage.changeVideoTitle:
			return SettingPage.setTitle(state, (action as IVideoSettingAO).title);
			break;

		case ActionTypes.Video.settingPage.setError:
			const { error } = action as IVideoSettingAO;
			return SettingPage.setError(state, error);
			break;
		
		case ActionTypes.Video.settingPage.changeCategories:
			return SettingPage.setCategories(state, (action as IVideoSettingAO).categories);
			break;

		case ActionTypes.Video.settingPage.changeRefFile:
			const { refFileName, refFilePath } = action as IVideoSettingAO;
			return SettingPage.setRefFile(state, refFileName, refFilePath);
			break;

		default:
			return state;
			break;
	}
};

const showList = (state, searchInput, videoList, pageInfo): IVideoReducer => {
	let newPage = 0;
	let reloadCursorMap: Map<number, string> = null;

	const oldResult = state.list;
	if (searchInput.cursor) {
		// setting map to get the cursor to reload page
		reloadCursorMap = new Map(oldResult.reloadCursorMap);

		if (searchInput.cursor.cursor) {
			newPage = oldResult.pageInfo.currentPage;

		} else if (searchInput.cursor.nextCursor) {
			newPage = oldResult.pageInfo.currentPage + 1;
			// set the last cursor as the reloadCursor
			reloadCursorMap.set(newPage, pageInfo.lastCursor);

			// remove total in pageInfo (because of ScrollingModel not have total => wrong paging if using it)
			delete pageInfo['total'];

		} else if (searchInput.cursor.prevCursor) {
			newPage = oldResult.pageInfo.currentPage - 1;

			// remove total in pageInfo (because of ScrollingModel not have total => wrong paging if using it)
			delete pageInfo['total'];
		}
	} else {
		reloadCursorMap = new Map();
	}

	return {
		...state,
		list: {
			...state.list,
			searchInput: searchInput,
			videoList,
			pageInfo: {
				...state.list.pageInfo,
				...pageInfo,
				currentPage: newPage
			},
			reloadCursorMap,
			isLoading: false,
		}
	};
};

const setSettingLoading = (state, isLoading = true): IVideoReducer => {
	return {
		...state,
		setting: {
			...state.setting,
			isLoading
		}
	};
};

const setListLoading = (state, isLoading = true): IVideoReducer => {
	return {
		...state,
		list: {
			...state.list,
			isLoading
		}
	};
};

const setVideoList = (state, videoList): IVideoReducer => {
	return {
		...state,
		list: {
			...state.list,
			videoList,
		}
	};
};

const setVideo = (state, action): IVideoReducer => {
	return {
		...state,
		setting: {
			...action.setting,
			isLoading: false
		}
	};
};

const setVideoError = (state, error): IVideoReducer => {
	const setting: IVideoSettingAO = state.setting;
	return {
		...state,
		setting: {
			...setting,
			error
		}
	};
};

const SettingPage = {
	setLoading: (state: IVideoReducer, isLoading): IVideoReducer => {
		return {
			...state,
			setting: {
				...state.setting,
				isLoading,
			},
		};
	},
	setTitle: (state: IVideoReducer, title) => {
		const setting: IVideoSettingAO = {
			...state.setting,
			title,
		};

		return {
			...state,
			setting,
		};
	},
	
	setCategories: (state: IVideoReducer, categories) => {
		const setting: IVideoSettingAO = {
			...state.setting,
			categories,
			categoryIds: categories.map((item, id) => Number(item.value.id)),
		};

		return {
			...state,
			setting,
		};
	},
	
	setRefFile: (state: IVideoReducer, refFileName, refFilePath) => {
		const setting: IVideoSettingAO = state.setting;
		return {
			...state,
			setting: {
				...setting,
				refFileName,
				refFilePath,
			},
		};
	},
	
	resetCreateOrUpdateForm: (state) => {
		return {
			...state,
			setting: {},
		};
	},
	loadInfo: (state: IVideoReducer, action) => {
		const setting: IVideoSettingAO = action.setting;
		return {
			...state,
			setting: {
				...setting,
				// sort process by order
				processes: setting?.processes?.sort(
					(a, b) => (a.order || 0) - (b.order || 0)
				),
			},
		};
	},
	setError: (state: IVideoReducer, error) => {
		const setting: IVideoSettingAO = state.setting;
		return {
			...state,
			setting: {
				...setting,
				error,
			},
		};
	},
	
};


export default handler;