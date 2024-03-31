import React, { Component, Fragment, RefObject } from 'react';
import tableCss from 'compDir/Theme/tableCss';
import { createStyles, Theme, withStyles } from '@material-ui/core/styles';
import MDIcon from '@mdi/react';
import { mdiDownload, mdiMenuUp, mdiMenuDown, mdiPlus } from '@mdi/js';
import { toast } from 'material-react-toastify';
import {
	Box, FormGroup, Grid, Typography, Checkbox, Radio, Paper, Card, 
	Table, TableBody, TableCell,
	TableContainer, TableHead, TablePagination, TableRow,
} from '@material-ui/core';

import Spinner from 'compDir/Spinner';
import Modal from 'compDir/Modal';
import {
	IGraphqlPageInfo, IVideo, IVideoSearchOpt,
} from 'interfaceDir';
import { DatePicker } from 'compDir/DateTimePicker';
import { Paging, VideoStatus, TIME_ZONE } from 'constDir';
import Utils from 'servDir/utils';

import moment from 'moment-timezone';
moment.tz.setDefault(TIME_ZONE);

import ColorButton from 'rootDir/components/Button';
import VideoItem from '../Item';
import SearchBar from '../SearchBar';
import VideoSetting from '../Setting';


export interface IStateToProps {
	videoList: Array<IVideo>
	pageInfo: IGraphqlPageInfo
	isLoading: boolean
	currentUser: any
}

export interface IDispatchToProps {
	initData(_comp: React.Component): any
	doSearch({ options, cursor, limit }): any
	doReloadVideo(videoId: number): any
	doPaging(pageNum: number): any
	updateLimit(limit): any
	loadSetting(modalTypeId: number, videoId: number): any
	changeEnabled(video: any, callback: any, options: any): any
	onStop(video: any, callback: any, options: any): any
}

interface IProps extends IDispatchToProps, IStateToProps {
	pauseInitData?: boolean
}

interface IState {
	firstLoading?: boolean
	searchBarRef: RefObject<any>
	firstLoading?: boolean

	settingModal: {
		id: number,
		title: string,
		label: string
		submitLabel: string,
	}
	reviewVideoLink: null,

	confirmBox: {
		type: '追加' | '編集' | '停止' | '再開',
		callback?(): any
	}

}
const useStyles = (theme: Theme) => createStyles({
	...tableCss,
	tableBody: {
		['& tr:nth-of-type(4n-1)']: {
			backgroundColor: theme.palette.action.hover
		}
	},
	modal: {
		margin: 0,
		position: 'absolute',
		top: 0
	},
});

export const ModalType = {
	create: {
		id: 1,
		title: 'Add Video',
		label: 'Add',
		submitLabel: 'Save',
	},
	update: {
		id: 2,
		title: 'Edit Video',
		label: 'Edit',
		submitLabel: 'Save',
	},
	closeModal: {
		id: 0,
		title: '',
		submitLabel: '',
	},
};

class Page extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
		firstLoading: true,
		videoId: 0,
		searchBarRef: React.createRef(),
	}

	//========================== Functions ==========================//

	gotToPage = (pageNum) => {
		const currentPage = this.props.pageInfo.currentPage;
		this.props.doPaging(pageNum, pageNum < currentPage);
	}

	onSearch = () => {
		this.props.doSearch({
			options: this.prepareSearchOptions(),
			limit: this.props.pageInfo.limit
		});
	}

	prepareSearchOptions = (searchOpts = this.state.searchBarRef.current?.state.searchOpts) => {
		const keyword = searchOpts.keyword;
		const status = searchOpts.status;
		const categories = searchOpts.categories;
		const startDate = searchOpts.startDate?.toISOString();
		const endDate = searchOpts.endDate?.toISOString();
		
		const options = {
			status: status,
		} as any;

		if (startDate?.length > 0) {
			options.startDate = startDate;
		}

		if (endDate?.length > 0) {
			options.endDate = endDate;
		}

		if (keyword?.length > 0) {
			options.keyword = keyword;
		}

		if (categories?.length > 0) {
			options.categories = categories;
		}

		return options;
	}

	handleClearSched = () => {
		this.props.loadSetting(this.state.settingModal.id, this.state.videoId);
	};

	openSettingModal = (modalTypeId, videoId = 0) => {
		let settingModal = ModalType.closeModal;
		switch (modalTypeId) {
			case ModalType.create.id:
				settingModal = ModalType.create;
				break;

			case ModalType.update.id:
				settingModal = ModalType.update;
				break;

			default:
				settingModal = ModalType.closeModal;
				break;
		}

		this.props.loadSetting(modalTypeId, videoId)
			.then(() => this.setState({
				settingModal,
				videoId
			}))
			.catch(error => toast.error(error.message));

	};

	closeModal = () => {
		this.setState({
			settingModal: null,
			videoId: 0,
		});
	};

	toggleReviewVideoModal = (video = null) => {
		this.setState({
			reviewVideo: video,
		});
	};

	handleSubmitVideo = async () => {
		try {
			this.props.onSubmitVideo()
				.then((result) => {
					if (result) {
						toast.success(`Videoの${this.state.settingModal.label}が成功しました`);
						this.closeModal();
					} else {
						toast.error(`Videoの${this.state.settingModal.label}が失敗しました.`);
					}
				})
				.catch(error => toast.error(error.message));
		} catch (err) {
			console.error(err);
		}
	};

	toggleVideoStatus = (video) => {
		const label = video.isEnabled ? 'Play' : 'Pause';

		this.openConfirmBox(label, () => {
			this.props.changeEnabled({
				id: video.id,
				isEnabled: video.isEnabled,
			}, (result) => {
				if (result) {
					toast.success(`${label} Video successfully`);
				} else {
					toast.error(`Cannot ${label} Video`);
				}
			}, this.state.searchOpts);
		});
	};

	onStop = (video) => {
		this.openConfirmBox('Delete', () => {
			this.props.onStop({
				id: video.id,
				isEnabled: false,
			}, (result) => {
				if (result) {
					toast.success(`Video has been stopped`);
				} else {
					toast.error(`Cannot stop Video`);
				}
			}, this.state.searchOpts);
		});
	};

	reviewVideo = (video) => {
		this.toggleReviewVideoModal(video);
	};

	openConfirmBox = (type, callback: any) => {
		this.setState({
			confirmBox: {
				type,
				callback
			}
		});
	}

	closeConfirmBox = (e, isConfirmed: boolean) => {
		const callback = this.state.confirmBox.callback;
		if (isConfirmed && typeof callback === 'function') {
			callback();
		}

		this.setState({
			confirmBox: null
		});
	}
	//========================== Lefe circle functions ==========================//
	componentDidMount() {
		// Load data for the first time load page --> trigger Only once-time 
		if (!this.props.pauseInitData) {
			this.props.initData(this);
		}
	}

	async componentDidUpdate(prevProps: IProps, prevState: IState, snapshot) {
		// init data on change page after clicked on menu --> Trigger multi-time
		if (this.props.pauseInitData != prevProps.pauseInitData) {
			this.props.initData(this);
		}
	}

	//========================== Render functions ==========================//
	public render(): React.ReactNode {
		const { classes } = this.props;
		const { exportOptions, confirmBox } = this.state;

		if (this.state.firstLoading != false) {
			return <Spinner title="Loading" />;
		}
		const searchOpts: IVideoSearchOpt = this.state.searchBarRef.current?.state.searchOpts;
		const searchTaskInput = searchOpts ? this.prepareSearchOptions(searchOpts) : {};

		return (
			<Fragment>
				<Grid container spacing={4} direction='column' wrap='nowrap'>
					<Grid item>
						<Grid container spacing={2} direction='column' wrap='nowrap'>
							<Grid item>
								<Grid container justify='space-between' alignItems='center'>
									<Typography component='div' variant="h6">Video Settings</Typography>
									<ColorButton variant='contained' onClick={() => this.openSettingModal(ModalType.create.id)}
										startIcon={<MDIcon size={'20px'} path={mdiPlus} />}>
										Add New Video
									</ColorButton>
								</Grid>
							</Grid>
							<Grid item>
								<Grid container justify='space-between' alignItems='flex-start'>
									<SearchBar ref={this.state.searchBarRef} onSearch={this.onSearch} />
								</Grid>
							</Grid>
						</Grid>
					</Grid>
					<Grid item>
						{
							this.props.isLoading != false ? (
								<Spinner title="Loading" />
							) : (
								<Paper elevation={0} >
									<TableContainer className={classes.tableContainer}>
										<Table>
											<TableHead>
												<TableRow>
													<TableCell style={{ minWidth: 65 }} align='center'><Box paddingLeft={2}>#ID</Box></TableCell>
													<TableCell><Box>Name</Box></TableCell>
													<TableCell><Box>File</Box></TableCell>
													<TableCell><Box>Category</Box></TableCell>
													<TableCell>Create time</TableCell>
													<TableCell>Status</TableCell>
													<TableCell align='center'><Box>Action</Box></TableCell>
													<TableCell></TableCell>
												</TableRow>
											</TableHead>
											<TableBody className={classes.tableBody}>
												{
													(this.props.videoList && this.props.videoList.length) ? this.props.videoList.map((video, index) => {
														return <VideoItem key={video.id} {...video}
															onClickEdit={() => { this.openSettingModal(ModalType.update.id, Number(video.id), index); }}
															reloadPage={() => this.props.doPaging(this.props.pageInfo.currentPage)}
															onChangeStatus={this.toggleVideoStatus}
															onStop={this.onStop}
															onReviewVideo={() => this.reviewVideo(video)}
														/>;
													}) : (
														<TableRow>
															<TableCell colSpan={12} align='center'>No information</TableCell>
														</TableRow>
													)
												}
											</TableBody>
										</Table>
										<TablePagination
											rowsPerPageOptions={Paging.rowPerPage.list}
											component="div"
											count={this.props.pageInfo.total}
											rowsPerPage={this.props.pageInfo.limit}
											page={this.props.pageInfo.currentPage}
											onChangePage={(e, pageNum) => this.gotToPage(pageNum)}
											onChangeRowsPerPage={(e) => this.props.updateLimit(e.target.value)}
											labelDisplayedRows={({ from, to, count }) => `${from} ~ ${to} / ${count !== -1 ? count : `${to} 以上`} items`}
											labelRowsPerPage="Page rows:"
										/>
									</TableContainer>

								</Paper>
							)
						}
					</Grid>
				</Grid>

				{
					this.state.settingModal /* || true */ && (
						<Modal maxWidth='xs' fullWidth
							title={this.state.settingModal.title}
							handleSubmit={async () => {
								if (await this.props.validateVideo()) {
									this.openConfirmBox(this.state.settingModal.label, () => {
										this.handleSubmitVideo();
									});
								}
							}}
							submitLabel={this.state.settingModal.submitLabel}
							handleClose={this.closeModal} closeLabel='Cancel'
							extBtn extBtnLabel='Reset' extBtnColor='red'
							handleExtBtn={() => this.handleClearSched()}
							justifyActionBtn='space-between'
							slide='down' custClasses={{ paper: classes.modal }}>
							<VideoSetting />
						</Modal>
					)
				}
				{
					this.state.reviewVideo && (
						<Modal maxWidth='md' 
							title={`Video: ${this.state.reviewVideo.title}`}
							
							handleClose={() => this.toggleReviewVideoModal()} closeLabel='Close'
							justifyActionBtn='space-between'
							slide='down' custClasses={{ paper: classes.modal }}>
							<video controls width="400">
								<source src={`/s3/${this.state.reviewVideo.refFilePath}`} type="video/mp4" />
							</video>
						</Modal>
					)
				}
				{
					confirmBox && (
						<Modal fullWidth divider={false} maxWidth='xs'
							title={`${confirmBox.type} confirmation`}
							content={`Are you sure to ${confirmBox.type} this file？`}
							handleClose={(e) => this.closeConfirmBox(e)}
							handleSubmit={(e) => this.closeConfirmBox(e, true)}
							submitLabel={confirmBox.type} closeLabel='Cancel'></Modal>
					)
				}

			</Fragment >
		);
	}
}
export default withStyles(useStyles)(Page);