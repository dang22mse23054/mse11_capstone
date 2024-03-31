import { IVideo, IVideoError } from 'interfaceDir';
import { IActionObj, IVideoSearchAO, IVideoSettingAO } from '../action-object';
import ActionTypes from './types';

export class VideoAction {
	public static SettingPage = {

		setVideoInfo: (setting: any): IActionObj => {
			return {
				type: ActionTypes.settingPage.setInfo,
				setting
			};
		},

		setVideoError: (error: IVideoError): IVideoSettingAO => {
			return {
				type: ActionTypes.settingPage.setError,
				error
			};
		},

		changeTitle: (title: string): IActionObj => {
			return {
				type: ActionTypes.settingPage.changeVideoTitle,
				title
			};
		},

		changeCategories: (categories: Array<number>): IVideoSettingAO => {
			return {
				type: ActionTypes.settingPage.changeCategories,
				categories,
			};
		},

		changeRefFile: (fileInfo: IFile): IVideoSettingAO => {
			return {
				type: ActionTypes.settingPage.changeRefFile,
				refFileName: fileInfo.fileName,
				refFilePath: fileInfo.filePath
			};
		},
	}

	public static ListPage = {

		setLoading: (isLoading = true): IActionObj => {
			return {
				type: ActionTypes.listPage.setIsLoading,
				isLoading
			};
		},

		showSearchResult: (searchInput, videoList: Array<IVideo>, pageInfo: any): IVideoSearchAO => {
			return {
				type: ActionTypes.listPage.search.show,
				searchInput,
				videoList,
				pageInfo
			};
		},
	}
}