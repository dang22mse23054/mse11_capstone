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
		case ActionTypes.Video.SHOW_VIDEO_LIST:
			const { searchInput, videoList, pageInfo } = (action as IVideoSearchAO);
			return showList(state, searchInput, videoList, pageInfo);
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


export default handler;