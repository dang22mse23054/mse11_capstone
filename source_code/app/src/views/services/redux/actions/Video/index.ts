import { IVideo, IVideoError } from 'interfaceDir';
import { IActionObj, IVideoSearchAO, IVideoSettingAO } from '../action-object';
import ActionTypes from './types';

export class VideoAction {
	public static SettingPage = {

		setVideoInfo: (setting: any): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_INFO,
				setting
			};
		},

		setVideoError: (error: IVideoError): IVideoSettingAO => {
			return {
				type: ActionTypes.SET_VIDEO_ERROR,
				error
			};
		},

		changeName: (name: string): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_NAME,
				name
			};
		},

		changeDeadline: (deadline: number): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_DEADLINE,
				deadline
			};
		},

		changeOwner: (owner: any): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_OWNER,
				owner
			};
		},

		changeProcesses: (processes: any): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_PROCESS,
				processes
			};
		},

		changeMentionUsers: (mentionUsers: any): IVideoSettingAO => {
			return {
				type: ActionTypes.SET_VIDEO_MENTION_USERS,
				mentionUsers,
			};
		},

		changeMentionContent: (mentionContent: string): IVideoSettingAO => {
			return {
				type: ActionTypes.SET_VIDEO_MENTION_CONTENT,
				mentionContent
			};
		},
	}

	public static ListPage = {

		setLoading: (isLoading = true): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_LIST_LOADING,
				isLoading
			};
		},

		showSearchResult: (searchInput, videoList: Array<IVideo>, pageInfo: any): IVideoSearchAO => {
			return {
				type: ActionTypes.SHOW_VIDEO_LIST,
				searchInput,
				videoList,
				pageInfo
			};
		},

		changeVideoList: (videoList: Array<any>): IActionObj => {
			return {
				type: ActionTypes.SET_VIDEO_LIST,
				videoList
			};
		}
	}
}