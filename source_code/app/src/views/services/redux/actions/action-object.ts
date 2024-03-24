// https://stackoverflow.com/questions/53312320/webpack-dev-server-ts-loader-not-reloads-when-interface-changed
import {
	IPagingObj,
	ICursorInput,
	IGraphqlPageInfo,
	ICAUserOption,
	IVideo,
	ICategory,
} from 'interfaceDir';

export interface IActionObj {
	type: string;
	obj?: any;
}

export interface IAuthActionObj extends IActionObj {
	authData?: any;
	isFirstLoading?: boolean;
}

// ------ Video ------ //
export interface IVideoActionObj extends IActionObj, IVideo {
	isLoading?: boolean;
}

export interface IVideoSettingAO extends IActionObj, IVideo {
	isLoading?: boolean;
	error?: IVideoError;
}

export interface IVideoSearchAO extends IActionObj, IPagingObj {
	searchInput?: IVideoSearchOpt;
	videoList?: Array<IVideo>;
	pageInfo: IGraphqlPageInfo;
	reloadCursorMap: Map<number, string>;
	isFirstLoading?: boolean;
	isLoading?: boolean;
	errMsg?: string;
}

// ------ Cateogry ------ //
export interface ICategoryAO extends IActionObj {
	initCategoryList?: Array<ICategory>;
  }