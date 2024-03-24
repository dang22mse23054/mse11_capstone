import React, { Component, Fragment, RefObject } from 'react';
import tableCss from 'compDir/Theme/tableCss';
import { createStyles, Theme, withStyles } from '@material-ui/core/styles';
import MDIcon from '@mdi/react';
import { mdiDownload, mdiMenuUp, mdiMenuDown } from '@mdi/js';
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
}

interface IProps extends IDispatchToProps, IStateToProps {
	pauseInitData?: boolean
}

interface IState {
	firstLoading?: boolean
	searchBarRef: RefObject<any>
	firstLoading?: boolean

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

class Page extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
		firstLoading: true,
		scheduleId: 0,
		searchBarRef: React.createRef(),
	}

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
									<Typography variant="h6">Video Settings</Typography>
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
			</Fragment >
		);
	}
}
export default withStyles(useStyles)(Page);