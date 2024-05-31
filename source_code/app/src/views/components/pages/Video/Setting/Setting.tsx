import React, { Component, Fragment } from 'react';
import clsx from 'clsx';

import { createStyles, Theme, withStyles } from '@material-ui/core/styles';

import FrequencyPicker from 'compDir/FrequencyPicker';
import { ICategoryOption, IMedia, IFile } from 'interfaceDir';
import commonCss from 'compDir/Theme/commonCss';
import CustomLabel from 'compDir/CustomLabel';
import CAUserTagInput from 'compDir/TagInput/CAUser';
import SingleLvTagInput from 'compDir/TagInput/SingleLevel';
import DropDownUser from 'compDir/DropDownUser';
import { ToggleButton, ToggleButtonGroup } from '@material-ui/lab';
import { DestinationTypes, ScheduleTypes, RequestChannel } from 'constDir';
import ClientTagInput from 'compDir/TagInput/Client';
import OrgTagInput from 'compDir/TagInput/Organization';
import Autocomplete from '@material-ui/lab/Autocomplete';
import ProcessSetting from 'compDir/ProcessSetting';
import ProcMasterTagInput from 'compDir/TagInput/ProcMaster';
import DropDownSelect from 'compDir/DropDownSelect';
import UploadButton from 'compDir/Button/Upload';
import { Status } from 'constDir';
import { Box, Grid, Typography, TextField, Chip, Tooltip } from '@material-ui/core';
import { IVideoSettingAO } from 'servDir/redux/actions';
import { mdiApplicationCog, mdiBullhornOutline, mdiCalendarClock, mdiCommentAccount, mdiCommentTextOutline, mdiMovieOpenOutline, mdiPackageUp } from '@mdi/js';

export interface IStateToProps {
	videoInfo: IVideoSettingAO
	largeCategories: Array<ICategoryOption>
	categoryList: Array<IMedia>
}

export interface IDispatchToProps {
	initData(_comp: React.Component): any

	// Setting
	changeCategories(categories, reason): any
	changeTitle(title): any
	changeRefFile(fileInfo: IFile): any
}

export interface IProps extends IDispatchToProps, IStateToProps {
	videoId?: number
	isClone?: boolean
}

interface IState {
	smallCategories: Array<ICategoryOption>
	selectOption: Array<any>
}

const useStyles = (theme: Theme) => createStyles({
	...commonCss,
	chip: {
		minWidth: theme.spacing(7)
	},
});


// When we need to access this component's state from the parent
// Should not define Functional Component because 'ref' attribute only available on Class Component
class Setting extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
		frequencyId: 1,
		files: [],
		caDashboards: [],
		media: [],
		smallCategories: [],
		selectOption: []
	}

	componentDidMount() {
		// Load data for the first time load page --> trigger Only once-time 
		if (!this.props.pauseInitData) {
			this.props.initData(this);
		}
	}

	changeVideoTitle = (event: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => {
		this.props.changeTitle(event.target.value);
	}

	public render(): React.ReactNode {
		const { videoInfo = {} as IVideoSettingAO, categoryList, classes } = this.props;
		const { smallCategories } = this.state;
		const { largeCategory, smallCategory, mediaIds = [] } = videoInfo;
		const isUpdate = Boolean(videoInfo?.id);
		const destType = videoInfo.destType || 1;
		const procMaster = videoInfo.procMaster ? videoInfo.procMaster[0] : undefined;

		const selectedProcChip = [{
			key: `${procMaster?.id}${procMaster?.name}`,
			value: {
				id: procMaster?.id,
				name: procMaster?.name,
				status: 0,
				dir: `${largeCategory?.value?.name}|${smallCategory?.value?.name}|${procMaster?.name}（#${procMaster?.id}）`
			}
		}];
		
		return (
			<Grid container spacing={2} alignContent='flex-start'>
				<Grid item xs={12}>
					<Grid container spacing={3} direction='column'>
						<Grid item>
							<Grid container alignItems='center' spacing={2}>
								<Grid item style={{ flex: 1 }}>
									<TextField fullWidth
										value={videoInfo.title || ''}
										onChange={this.changeVideoTitle}
										label="Title"
										error={Boolean(videoInfo.error?.title)}
										helperText={videoInfo.error?.title} />
								</Grid>
							</Grid>
						</Grid>
					</Grid>
				</Grid>
				<Grid item xs={12}>
					<Grid container spacing={3} alignContent='flex-start'>
						<Grid item xs={12}>
							<SingleLvTagInput label='Select categories' limitTags={2} size='small'
								onChange={this.props.changeCategories}
								options={categoryList}
								value={videoInfo.categories ? videoInfo.categories : []}
								isError={Boolean(videoInfo.error?.categories)}
								helperText={videoInfo.error?.categories}
							/>
						</Grid>
					</Grid>
				</Grid>
				<Grid item xs={12}></Grid>
				<Grid item xs={12}>
					<Grid container spacing={2} style={{ position: 'relative' }}>
						<Grid item xs={6} style={{ flexBasis: '50%' }}>
							<Grid container spacing={2} direction='column'>
								
								<Grid item container>
									<Typography variant='subtitle2'>
										<Box color='text.secondary' fontWeight='fontWeightBold' marginBottom={1}>
											<CustomLabel value={'Video file'} icon={mdiMovieOpenOutline} />
										</Box>
									</Typography>
									<Grid container spacing={2} alignItems='center'>
										<Grid item style={{ flex: 1 }}>
											<UploadButton size='small' variant='outlined' btnColor='secondary'
												dispType='chip'
												maxWidth={300}
												accept='*.*'
												fileInfo={{
													fileName: videoInfo.refFileName,
													filePath: videoInfo.refFilePath
												}}
												onSuccess={this.props.changeRefFile}
												onDelete={this.props.changeRefFile}

											>Upload video</UploadButton>
										</Grid>
									</Grid>
								</Grid>
								
							</Grid>
						</Grid>
						
					</Grid>
				</Grid>

			</Grid>
		);
	}
}

export default withStyles(useStyles)(Setting);